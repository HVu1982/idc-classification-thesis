import streamlit as st
import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import traceback
import datetime
import json
import pandas as pd
from pathlib import Path
import gdown
from streamlit_image_zoom import image_zoom
import shutil

# ============================================================
# 1. THIẾT LẬP MÔI TRƯỜNG & IMPORT CONFIG
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import config
except ImportError:
    st.error("❌ Lỗi: Không tìm thấy file config.py.")
    st.stop()

sys.path.append(str(config.SRC_DIR))

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {background-color: #f8f9fa;}
        div[data-testid="stMetricValue"] {font-size: 1.1rem; font-weight: bold;}
        div[data-testid="stDataFrame"] {font-size: 0.85rem;}
        .block-container {padding-top: 2rem;}
        .author-box {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #1976d2;
            font-size: 0.9rem;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. HÀM TẢI MODEL (TÁCH BIỆT UI VÀ LOGIC)
# ============================================================
MODEL_DRIVE_ID = "1Ruvjg57t-JLoP1QcWK_I8UzcFuUFjCnN"  # ID file .pth của bạn

def check_and_download_model():
    """
    Hàm kiểm tra và tải model.
    Hàm này KHÔNG được cache để đảm bảo logic kiểm tra file luôn chạy.
    """
    if not config.MODEL_PATH.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
        output = str(config.MODEL_PATH)
        try:
            # Tải file im lặng, xử lý UI ở bên ngoài
            gdown.download(url, output, quiet=False)
            return True
        except Exception:
            return False
    return True

# --- 3. CLASS & CORE FUNCTIONS ---
class WSIPatchDataset(Dataset):
    def __init__(self, image, coords, patch_size=50, transform=None):
        self.image = image; self.coords = coords; self.patch_size = patch_size; self.transform = transform
    def __len__(self): return len(self.coords)
    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch = self.image[y : y + self.patch_size, x : x + self.patch_size]
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        if self.transform: patch = self.transform(patch)
        return patch

@st.cache_resource
def load_model_cached(device_name):
    """
    Hàm load model ĐƯỢC CACHE.
    QUAN TRỌNG: Không được chứa bất kỳ lệnh st. (streamlit UI) nào ở đây.
    """
    device = torch.device(device_name)
    try:
        from src.model_hybrid1 import CNNDeiTSmall
        model = CNNDeiTSmall(**config.MODEL_PARAMS)
        
        # Kiểm tra file lần nữa cho chắc chắn
        if not config.MODEL_PATH.exists():
            return None
            
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None

def generate_tissue_mask(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 230]); upper_white = np.array([180, 25, 255]) 
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    tissue_mask = cv2.bitwise_not(mask_white)
    kernel = np.ones((5,5), np.uint8)
    tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=1)
    return tissue_mask

def run_inference(model, image_array, device, threshold, batch_size, max_patches, stride, progress_bar):
    h, w = image_array.shape[:2]
    patch_size = config.PATCH_SIZE
    
    tissue_mask = generate_tissue_mask(image_array)
    coords = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            mask_roi = tissue_mask[y : y + patch_size, x : x + patch_size]
            if cv2.countNonZero(mask_roi) / (patch_size**2) > 0.05:
                coords.append((y, x))
    
    if not coords: return None, None, {"cancer_percentage": 0.0}

    total_found = len(coords)
    if max_patches > 0 and total_found > max_patches:
        coords = coords[:max_patches]
        # Bỏ st.toast ở đây để an toàn tuyệt đối cho luồng tính toán
        # st.toast(f"⚡ Demo Mode: {max_patches}/{total_found} patches", icon="🚀")

    transform = T.Compose([T.ToPILImage(), T.Resize(config.MODEL_PARAMS['img_size']), T.ToTensor(), T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)])
    dataset = WSIPatchDataset(image_array, coords, patch_size, transform)
    num_workers = 0 if os.name == 'nt' else config.NUM_WORKERS
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    all_confidences = []
    
    with torch.no_grad():
        batch_start_idx = 0
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            probs_np = probs.cpu().numpy()
            
            all_confidences.extend(probs_np)
            
            current_batch_size = len(probs_np)
            batch_coords = coords[batch_start_idx : batch_start_idx + current_batch_size]
            
            for (y, x), p in zip(batch_coords, probs_np):
                prob_map[y:y+patch_size, x:x+patch_size] += p
                count_map[y:y+patch_size, x:x+patch_size] += 1
            
            batch_start_idx += current_batch_size

            if progress_bar: progress_bar.progress((i+1)/len(loader), text=f"Processing batch {i+1}/{len(loader)}...")

    avg_heatmap = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map!=0)

    # --- TẠO OVERLAY ---
    overlay = image_array.copy()
    binary_mask = (avg_heatmap >= threshold).astype(np.uint8)
    kernel_smooth = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    red_layer = np.zeros_like(overlay)
    red_layer[:] = [255, 0, 0]
    
    mask_indices = binary_mask == 1
    if np.any(mask_indices):
        overlay[mask_indices] = cv2.addWeighted(overlay[mask_indices], 0.6, red_layer[mask_indices], 0.4, 0)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (200, 0, 0), 2)

    total_tissue_pixels = np.count_nonzero(count_map)
    cancer_pixels = np.count_nonzero(binary_mask)
    cancer_percentage = round((cancer_pixels / total_tissue_pixels) * 100, 2) if total_tissue_pixels > 0 else 0.0

    stats = {
        "total_patches": len(coords), "original_patches": total_found, 
        "cancer_patches": int(np.sum(np.array(all_confidences) >= threshold)),
        "cancer_percentage": cancer_percentage,
        "max_confidence": round(float(np.max(all_confidences)), 4) if all_confidences else 0
    }
    return overlay, avg_heatmap, stats

# ============================================================
# 4. GIAO DIỆN CHÍNH (MAIN)
# ============================================================
def main():
    if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
    if 'history' not in st.session_state: st.session_state.history = []

    # === SIDEBAR ===
    with st.sidebar:
        if config.LOGO_PATH.exists(): st.image(str(config.LOGO_PATH), width=120)
        
        desc_html = config.APP_DESCRIPTION.strip().replace('\n', '<br>')
        st.markdown(f'<div class="author-box">{desc_html}</div>', unsafe_allow_html=True)

        st.header("⚙️ Cấu hình")
        dev_show = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"Thiết bị: **{dev_show}**")

        with st.expander("🛠️ Chi tiết & Tối ưu", expanded=False):
            st.markdown(f"**Hybrid CNN-DeiT** (Patches: {config.PATCH_SIZE}px)")
            if config.MODEL_VIZ_PATH.exists(): st.image(str(config.MODEL_VIZ_PATH), caption="Kiến trúc", use_column_width=True)
            
            ui_max_patches = st.slider("Giới hạn Patch (Demo)", 0, 5000, 0, 100,
                help="0 = Không giới hạn. Đặt số thấp (vd: 500) để demo nhanh hơn.")
            
            st.markdown("---")
            ui_stride = st.select_slider(
                "Độ mịn (Stride)", 
                options=[10, 25, 50], 
                value=25,
                help="10: Rất mịn (Chậm). 25: Mịn vừa. 50: Nhanh (Thô)."
            )

        ui_threshold = st.slider("Ngưỡng (Threshold)", 0.0, 1.0, config.CONFIDENCE_THRESHOLD, 0.05)
        ui_batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=3 if config.DEVICE=="cuda" else 1)

        if st.session_state.history:
            st.markdown("---")
            st.subheader("🕒 Lịch sử phiên")
            st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, height=150)

        st.markdown("---")
        with st.expander("📊 Công cụ Báo cáo", expanded=False):
            if st.button("📑 Tổng hợp CSV & Xem", use_container_width=True):
                results_dir = config.BASE_DIR / "results"
                csv_files = list(results_dir.glob("stats_*.csv")) if results_dir.exists() else []
                if not csv_files:
                    st.warning("Chưa có dữ liệu.")
                else:
                    try:
                        df_list = [pd.read_csv(f) for f in csv_files if "summary" not in f.name]
                        if df_list:
                            combined_df = pd.concat(df_list, ignore_index=True)
                            if 'timestamp' in combined_df.columns: combined_df = combined_df.sort_values(by='timestamp', ascending=False)
                            summary_path = results_dir / "summary_report.csv"
                            combined_df.to_csv(summary_path, index=False)
                            st.success(f"Đã gộp {len(df_list)} file!")
                            
                            def highlight(val): return 'background-color: #ffcccc' if isinstance(val, (int, float)) and val >= config.DANGER_THRESHOLD_PERCENT else ''
                            cols = [c for c in ['image_name', 'cancer_percentage', 'max_confidence', 'timestamp'] if c in combined_df.columns]
                            st.dataframe(combined_df[cols].style.map(highlight, subset=['cancer_percentage'] if 'cancer_percentage' in combined_df else None), hide_index=True)
                            
                            with open(summary_path, "rb") as f: st.download_button("⬇️ Tải file CSV", f, "summary_report.csv", "text/csv")
                    except Exception as e: st.error(f"Lỗi: {e}")

            if st.button("🗑️ Xóa toàn bộ lịch sử", type="primary"):
                results_dir = config.BASE_DIR / "results"
                if results_dir.exists():
                    shutil.rmtree(results_dir); results_dir.mkdir()
                    st.session_state.history = []; st.session_state.analysis_result = None
                    st.rerun()
        st.caption("© 2026 Vũ Hữu Hoàng")

    # === MAIN CONTENT ===
    st.title(config.APP_TITLE)
    st.write("---")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1. Chọn dữ liệu")
        input_method = st.radio("Nguồn:", ["Tải ảnh lên", "Dùng ảnh mẫu"], horizontal=True)
        uploaded_file = None; current_img_name = ""

        if input_method == "Tải ảnh lên":
            uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image_pil = Image.open(uploaded_file).convert('RGB')
                current_img_name = uploaded_file.name
        elif hasattr(config, 'SAMPLE_IMAGES'):
            sample_choice = st.selectbox("Mẫu:", list(config.SAMPLE_IMAGES.keys()))
            sample_path = config.SAMPLE_IMAGES[sample_choice]
            if sample_path.exists():
                image_pil = Image.open(sample_path).convert('RGB')
                current_img_name = sample_path.name
                class Mock: name=current_img_name
                uploaded_file = Mock()

        if 'image_pil' in locals() and image_pil:
            image_array = np.array(image_pil)
            st.image(image_pil, caption=f"Ảnh: {current_img_name}", use_column_width=True)
            analyze = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)
        else: analyze = False

    with col2:
        st.subheader("2. Kết quả")
        if analyze and image_pil:
            # --- BƯỚC 1: KIỂM TRA VÀ TẢI MODEL (NẾU CHƯA CÓ) ---
            # Bước này thực hiện ở ngoài hàm cache để có thể vẽ UI (spinner/toast)
            if not config.MODEL_PATH.exists():
                with st.spinner("⏳ Đang tải Model từ Cloud về máy chủ... (Vui lòng đợi 1-2 phút)"):
                    success = check_and_download_model()
                    if success:
                        st.toast("✅ Đã tải Model thành công!", icon="☁️")
                    else:
                        st.error("❌ Không thể tải model. Kiểm tra mạng hoặc link Drive.")
                        st.stop()

            # --- BƯỚC 2: LOAD MODEL (CACHED) ---
            # Gọi hàm cache thuần túy (không UI)
            progress = st.progress(0, text="Khởi tạo Model...")
            try:
                run_device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_model_cached(run_device)
                
                if model is None:
                    st.error("❌ Lỗi: Model file bị hỏng hoặc không tương thích.")
                else:
                    # --- BƯỚC 3: CHẠY SUY LUẬN ---
                    if ui_max_patches > 0:
                        st.toast(f"⚡ Chế độ Demo: Giới hạn {ui_max_patches} patches", icon="🚀")
                        
                    overlay, heatmap, stats = run_inference(
                        model, image_array, run_device, 
                        ui_threshold, ui_batch_size, ui_max_patches, ui_stride, progress
                    )
                    progress.empty()
                    
                    # Lưu kết quả vào Session
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.analysis_result = {'overlay': overlay, 'heatmap': heatmap, 'stats': stats, 'filename': current_img_name, 'timestamp': ts}
                    st.session_state.history.insert(0, {"Time": datetime.datetime.now().strftime("%H:%M"), "File": current_img_name, "Risk": f"{stats['cancer_percentage']}%"})
            except Exception as e:
                st.error("Lỗi hệ thống trong quá trình xử lý.")
                st.code(traceback.format_exc())

        # --- HIỂN THỊ KẾT QUẢ TỪ SESSION ---
        res = st.session_state.analysis_result
        if res and res.get('filename') == current_img_name:
            overlay, heatmap, stats, ts = res['overlay'], res['heatmap'], res['stats'], res['timestamp']
            
            t1, t2 = st.tabs(["🔍 Soi vùng bệnh", "🌡️ Heatmap"])
            
            hm_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
            hm_color = cv2.cvtColor(cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            blend = cv2.addWeighted(image_array, 0.6, hm_color, 0.4, 0)

            with t1:
                st.caption("👉 Di chuột để phóng to:")
                image_zoom(Image.fromarray(overlay), mode="mousemove", size=700, zoom_factor=3, keep_aspect_ratio=True)

            with t2: 
                st.image(blend, caption="Mức độ tin cậy (Heatmap)", use_column_width=True)
            
            # --- LƯU FILE KẾT QUẢ ---
            r_dir = config.BASE_DIR / "results"
            r_dir.mkdir(exist_ok=True)
            p_csv = r_dir / f"stats_{ts}.csv"
            
            if not p_csv.exists():
                try:
                    cv2.imwrite(str(r_dir/f"overlay_{ts}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(r_dir/f"heatmap_{ts}.png"), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
                    s_csv = stats.copy(); s_csv.update({'timestamp': ts, 'image_name': current_img_name, 'stride': ui_stride})
                    pd.DataFrame([s_csv]).to_csv(p_csv, index=False)
                    with open(r_dir/f"stats_{ts}.json", "w") as f: json.dump(stats, f, indent=2)
                except: pass

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng Patch", stats['total_patches'])
            c2.metric("IDC Patch", stats['cancer_patches'])
            clr = "inverse" if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT else "normal"
            c3.metric("Tỷ lệ bệnh", f"{stats['cancer_percentage']}%", delta_color=clr)
            c4.metric("Max Conf", stats['max_confidence'])

            if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT: st.error(f"🚨 NGUY CƠ CAO ({stats['cancer_percentage']}%)")
            else: st.success("✅ AN TOÀN")

            st.write("---")
            st.markdown("##### 📥 Tải kết quả về máy")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                with open(r_dir/f"overlay_{ts}.png", "rb") as f: st.download_button("🖼️ Ảnh Overlay", f, f"overlay_{ts}.png", "image/png")
            with d2:
                with open(r_dir/f"heatmap_{ts}.png", "rb") as f: st.download_button("🌡️ Ảnh Heatmap", f, f"heatmap_{ts}.png", "image/png")
            with d3:
                with open(r_dir/f"stats_{ts}.json", "rb") as f: st.download_button("📄 JSON Stats", f, f"stats_{ts}.json", "application/json")
            with d4:
                with open(p_csv, "rb") as f: st.download_button("📊 CSV Stats", f, p_csv.name, "text/csv")

            if st.button("🔄 Reset / Ca mới", type="secondary", use_container_width=True):
                st.session_state.analysis_result = None; st.rerun()

if __name__ == "__main__":
    main()
