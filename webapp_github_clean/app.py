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
# 2. TỰ ĐỘNG TẢI MODEL TỪ GOOGLE DRIVE
# ============================================================
MODEL_DRIVE_ID = "1Ruvjg57t-JLoP1QcWK_I8UzcFuUFjCnN"  # ⚠️ Thay ID file .pth của bạn vào đây

@st.cache_resource
def download_model_from_drive():
    if not config.MODEL_PATH.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
        output = str(config.MODEL_PATH)
        st.toast("⏳ Đang tải Model từ Cloud...", icon="☁️")
        try:
            gdown.download(url, output, quiet=False)
            st.success("✅ Tải Model thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi tải model: {e}")
            st.stop()

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
def load_model(device_name):
    download_model_from_drive()
    device = torch.device(device_name)
    try:
        from src.model_hybrid1 import CNNDeiTSmall
        model = CNNDeiTSmall(**config.MODEL_PARAMS)
        if not config.MODEL_PATH.exists(): return None
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e: return None

def generate_tissue_mask(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 230]); upper_white = np.array([180, 25, 255]) 
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    tissue_mask = cv2.bitwise_not(mask_white)
    kernel = np.ones((5,5), np.uint8)
    tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=1)
    return tissue_mask

def run_inference(model, image_array, device, threshold, batch_size, max_patches, progress_bar):
    h, w = image_array.shape[:2]
    patch_size = config.PATCH_SIZE; stride = config.STRIDE
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
        st.toast(f"⚡ Demo Mode: {max_patches}/{total_found} patches", icon="🚀")

    transform = T.Compose([T.ToPILImage(), T.Resize(config.MODEL_PARAMS['img_size']), T.ToTensor(), T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)])
    dataset = WSIPatchDataset(image_array, coords, patch_size, transform)
    num_workers = 0 if os.name == 'nt' else config.NUM_WORKERS
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    predictions, confidences = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predictions.extend((probs >= threshold).int().cpu().numpy())
            confidences.extend(probs.cpu().numpy())
            if progress_bar: progress_bar.progress((i+1)/len(loader), text=f"Processing batch {i+1}/{len(loader)}...")

    # --- TẠO OVERLAY LIỀN MẠCH & NHẠT HƠN ---
    heatmap = np.zeros((h, w), dtype=np.float32)
    overlay_mask = np.zeros((h, w), dtype=np.uint8) # Mask nhị phân
    cancer_count = 0
    
    for (y, x), pred, conf in zip(coords, predictions, confidences):
        heatmap[y:y+patch_size, x:x+patch_size] = conf
        if pred == 1:
            cancer_count += 1
            # Vẽ ô đặc full size (không trừ hao gap) -> Liền mạch
            cv2.rectangle(overlay_mask, (x, y), (x+patch_size, y+patch_size), 1, -1)

    # Tạo lớp màu đỏ từ mask
    # Chỉ tô đỏ những chỗ mask=1
    color_layer = np.zeros_like(image_array)
    color_layer[overlay_mask == 1] = [255, 0, 0] # Màu đỏ (RGB)

    # Blend: Ảnh gốc 70% + Lớp màu 30% -> Nhạt hơn, dễ nhìn tế bào bên dưới
    # (Phiên bản trước là 60/40, giờ giảm xuống 30% cho dịu mắt)
    overlay = image_array.copy()
    mask_indices = overlay_mask == 1
    if np.any(mask_indices):
        overlay[mask_indices] = cv2.addWeighted(image_array[mask_indices], 0.7, color_layer[mask_indices], 0.3, 0)
            
    stats = {
        "total_patches": len(coords), "original_patches": total_found, "cancer_patches": cancer_count,
        "cancer_percentage": round((cancer_count/len(coords))*100, 2),
        "max_confidence": round(float(np.max(confidences)), 4) if confidences else 0
    }
    return overlay, heatmap, stats

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

        with st.expander("🛠️ Chi tiết Mô hình", expanded=False):
            st.markdown(f"**Hybrid CNN-DeiT** (Patches: {config.PATCH_SIZE}px)")
            if config.MODEL_VIZ_PATH.exists(): st.image(str(config.MODEL_VIZ_PATH), caption="Kiến trúc", use_column_width=True)
            ui_max_patches = st.slider("Giới hạn Patch (Demo)", 0, 5000, 0, 100)

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
            progress = st.progress(0, text="Khởi tạo...")
            try:
                run_device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_model(run_device)
                if model:
                    overlay, heatmap, stats = run_inference(model, image_array, run_device, ui_threshold, ui_batch_size, ui_max_patches, progress)
                    progress.empty()
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.analysis_result = {'overlay': overlay, 'heatmap': heatmap, 'stats': stats, 'filename': current_img_name, 'timestamp': ts}
                    st.session_state.history.insert(0, {"Time": datetime.datetime.now().strftime("%H:%M"), "File": current_img_name, "Risk": f"{stats['cancer_percentage']}%"})
            except Exception as e: st.error("Lỗi hệ thống."); st.code(traceback.format_exc())

        res = st.session_state.analysis_result
        if res and res.get('filename') == current_img_name:
            overlay, heatmap, stats, ts = res['overlay'], res['heatmap'], res['stats'], res['timestamp']
            
            # --- TABS HIỂN THỊ CẢI TIẾN ---
            t1, t2 = st.tabs(["🔍 Vùng tổn thương", "🌡️ Heatmap"])
            
            hm_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
            hm_color = cv2.cvtColor(cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            blend = cv2.addWeighted(image_array, 0.6, hm_color, 0.4, 0)

            # Tab 1: So sánh Gốc vs Dự đoán (Dùng st.image mặc định để có tính năng phóng to)
            with t1:
                st.info("💡 Mẹo: Nhấn vào mũi tên ⤢ ở góc trên bên phải ảnh để xem toàn màn hình và phóng to chi tiết.")
                st.image(overlay, caption="Phát hiện IDC (Viền đỏ)", use_column_width=True)

            # Tab 2: Heatmap
            with t2: 
                st.image(blend, caption="Bản đồ nhiệt thể hiện độ tin cậy", use_column_width=True)
            
            # Lưu file
            r_dir = config.BASE_DIR / "results"
            r_dir.mkdir(exist_ok=True)
            p_csv = r_dir / f"stats_{ts}.csv"
            
            if not p_csv.exists():
                try:
                    cv2.imwrite(str(r_dir/f"overlay_{ts}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(r_dir/f"heatmap_{ts}.png"), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
                    s_csv = stats.copy(); s_csv.update({'timestamp': ts, 'image_name': current_img_name})
                    pd.DataFrame([s_csv]).to_csv(p_csv, index=False)
                    with open(r_dir/f"stats_{ts}.json", "w") as f: json.dump(stats, f, indent=2)
                except: pass

            # Metrics
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng Patch", stats['total_patches'])
            c2.metric("IDC Patch", stats['cancer_patches'])
            clr = "inverse" if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT else "normal"
            c3.metric("Tỷ lệ bệnh", f"{stats['cancer_percentage']}%", delta_color=clr)
            c4.metric("Max Conf", stats['max_confidence'])

            if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT: st.error(f"🚨 NGUY CƠ CAO ({stats['cancer_percentage']}%)")
            else: st.success("✅ AN TOÀN")

            # --- NÚT TẢI VỀ (ĐÃ KHÔI PHỤC) ---
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

