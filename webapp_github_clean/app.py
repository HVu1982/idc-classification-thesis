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

# --- 1. SETUP & IMPORT CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import config
except ImportError:
    st.error("❌ Không tìm thấy file config.py.")
    st.stop()

sys.path.append(str(config.SRC_DIR))

# --- 2. PAGE CONFIG ---
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {background-color: #f8f9fa;}
        div[data-testid="stMetricValue"] {font-size: 1.2rem; font-weight: bold;}
        .block-container {padding-top: 2rem;}
        div[data-testid="stDataFrame"] {font-size: 0.85rem;}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 📥 TỰ ĐỘNG TẢI MODEL
# ============================================================
MODEL_DRIVE_ID = "1AbC...XYZ_ID_CUA_BAN" 

@st.cache_resource
def download_model_from_drive():
    if not config.MODEL_PATH.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
        output = str(config.MODEL_PATH)
        st.toast("⏳ Đang tải Model từ Cloud (Lần đầu)...", icon="☁️")
        try:
            gdown.download(url, output, quiet=False)
            st.success("✅ Tải Model thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi tải model: {e}")
            st.stop()

# --- 3. CLASS & CORE FUNCTIONS ---

class WSIPatchDataset(Dataset):
    def __init__(self, image, coords, patch_size=50, transform=None):
        self.image = image
        self.coords = coords
        self.patch_size = patch_size
        self.transform = transform
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
        if not config.MODEL_PATH.exists():
            st.error(f"❌ File model không tồn tại: {config.MODEL_PATH}")
            return None
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Lỗi khởi tạo Model: {e}")
        return None

def generate_tissue_mask(img_rgb):
    # Logic lọc nền tối ưu v5.0
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 230])
    upper_white = np.array([180, 25, 255]) 
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    tissue_mask = cv2.bitwise_not(mask_white)
    kernel = np.ones((5,5), np.uint8)
    tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=1)
    return tissue_mask

def run_inference(model, image_array, device, threshold, batch_size, max_patches, progress_bar):
    h, w = image_array.shape[:2]
    patch_size = config.PATCH_SIZE
    stride = config.STRIDE
    
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
        st.toast(f"⚡ Giới hạn xử lý: {max_patches}/{total_found} patches", icon="🚀")

    transform = T.Compose([
        T.ToPILImage(), T.Resize(config.MODEL_PARAMS['img_size']), T.ToTensor(),
        T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
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
            if progress_bar:
                progress_bar.progress((i + 1) / len(loader), text=f"Processing batch {i+1}/{len(loader)}...")

    # KỸ THUẬT GRID PADDING (VẼ Ô VUÔNG TÁCH RỜI)
    heatmap = np.zeros((h, w), dtype=np.float32)
    overlay_layer = image_array.copy() # Layer riêng để vẽ màu
    cancer_count = 0
    gap = 2 # Khoảng hở giữa các ô (pixel)

    for (y, x), pred, conf in zip(coords, predictions, confidences):
        heatmap[y:y+patch_size, x:x+patch_size] = conf
        if pred == 1:
            cancer_count += 1
            # Vẽ hình chữ nhật nhỏ hơn patch một chút để tạo khe hở
            start_pt = (x + gap, y + gap)
            end_pt = (x + patch_size - gap, y + patch_size - gap)
            # Vẽ màu đỏ đặc (-1)
            cv2.rectangle(overlay_layer, start_pt, end_pt, (255, 0, 0), -1)

    # Blend màu đỏ vào ảnh gốc (Transparency 40%)
    overlay = cv2.addWeighted(image_array, 0.6, overlay_layer, 0.4, 0)
            
    stats = {
        "total_patches": len(coords), "original_patches": total_found,
        "cancer_patches": cancer_count,
        "cancer_percentage": round((cancer_count / len(coords)) * 100, 2),
        "max_confidence": round(float(np.max(confidences)), 4) if confidences else 0
    }
    return overlay, heatmap, stats

# --- 4. MAIN UI ---
def main():
    if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
    if 'history' not in st.session_state: st.session_state.history = []

    # === SIDEBAR ===
    with st.sidebar:
        if config.LOGO_PATH and config.LOGO_PATH.exists():
            st.image(str(config.LOGO_PATH), width=120)
        
        st.header("⚙️ Cấu hình")
        dev_show = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"Thiết bị: **{dev_show}**")

        with st.expander("🛠️ Chi tiết Mô hình", expanded=False):
            st.markdown(f"**Hybrid CNN-DeiT** (Patches: {config.PATCH_SIZE}px)")
            if hasattr(config, 'MODEL_VIZ_PATH') and config.MODEL_VIZ_PATH.exists():
                st.image(str(config.MODEL_VIZ_PATH), caption="Kiến trúc đề xuất", use_column_width=True)
            ui_max_patches = st.slider("Giới hạn Patch (Demo)", 0, 5000, 0, 100)

        ui_threshold = st.slider("Ngưỡng (Threshold)", 0.0, 1.0, config.CONFIDENCE_THRESHOLD, 0.05)
        default_bs = 3 if config.DEVICE == "cuda" else 1
        ui_batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=default_bs)

        if st.session_state.history:
            st.markdown("---")
            st.subheader("🕒 Lịch sử phiên")
            st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, height=150)

        # --- CÔNG CỤ BÁO CÁO (ĐÃ CẬP NHẬT HIỂN THỊ TRỰC TIẾP) ---
        st.markdown("---")
        with st.expander("📊 Công cụ Báo cáo", expanded=False):
            if st.button("📑 Tổng hợp CSV & Xem", use_container_width=True):
                results_dir = config.BASE_DIR / "results"
                csv_files = list(results_dir.glob("stats_*.csv"))
                if not csv_files:
                    st.warning("Chưa có dữ liệu.")
                else:
                    try:
                        df_list = [pd.read_csv(f) for f in csv_files if "summary" not in f.name]
                        if df_list:
                            combined_df = pd.concat(df_list, ignore_index=True)
                            if 'timestamp' in combined_df.columns:
                                combined_df = combined_df.sort_values(by='timestamp', ascending=False)
                            
                            summary_path = results_dir / "summary_report.csv"
                            combined_df.to_csv(summary_path, index=False)
                            
                            st.success(f"Đã gộp {len(df_list)} file!")
                            
                            # --- HIỂN THỊ BẢNG NGAY TẠI ĐÂY ---
                            st.markdown("##### Bảng kết quả tổng hợp:")
                            def highlight_risk(val):
                                color = '#ffcccc' if isinstance(val, (int, float)) and val >= config.DANGER_THRESHOLD_PERCENT else ''
                                return f'background-color: {color}'
                            
                            if 'cancer_percentage' in combined_df.columns:
                                st.dataframe(
                                    combined_df.style.map(highlight_risk, subset=['cancer_percentage'])
                                               .format({"cancer_percentage": "{:.2f}%", "max_confidence": "{:.4f}"}),
                                    use_container_width=True, hide_index=True
                                )
                            else:
                                st.dataframe(combined_df, use_container_width=True)
                            
                            with open(summary_path, "rb") as f:
                                st.download_button("⬇️ Tải file CSV", f, "summary_report.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

        st.caption("© 2026 Vũ Hữu Hoàng")

    # === MAIN CONTENT ===
    st.title(config.APP_TITLE)
    st.write("---")
    
    col1, col2 = st.columns([1, 1.5])

    # --- CỘT TRÁI ---
    with col1:
        st.subheader("1. Chọn dữ liệu")
        input_method = st.radio("Nguồn ảnh:", ["Tải ảnh lên", "Dùng ảnh mẫu (Demo)"], horizontal=True)
        uploaded_file = None
        current_img_name = ""

        if input_method == "Tải ảnh lên":
            uploaded_file = st.file_uploader("Upload H&E Image", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image_pil = Image.open(uploaded_file).convert('RGB')
                current_img_name = uploaded_file.name
        else:
            if hasattr(config, 'SAMPLE_IMAGES'):
                sample_choice = st.selectbox("Chọn ca bệnh mẫu:", list(config.SAMPLE_IMAGES.keys()))
                sample_path = config.SAMPLE_IMAGES[sample_choice]
                if sample_path.exists():
                    image_pil = Image.open(sample_path).convert('RGB')
                    current_img_name = sample_path.name
                else:
                    st.error("⚠️ File ảnh mẫu không tồn tại.")
                    image_pil = None

        if 'image_pil' in locals() and image_pil:
            image_array = np.array(image_pil)
            st.image(image_pil, caption=f"Ảnh đầu vào: {current_img_name}", use_column_width=True)
            analyze_trigger = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)
        else:
            analyze_trigger = False
            st.info("👈 Vui lòng chọn ảnh để bắt đầu.")

    # --- CỘT PHẢI ---
    with col2:
        st.subheader("2. Kết quả & Soi chi tiết")
        
        if analyze_trigger and image_pil:
            progress_bar = st.progress(0, text="Khởi tạo mô hình...")
            try:
                run_device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_model(run_device)
                if model:
                    overlay, heatmap, stats = run_inference(
                        model, image_array, run_device, 
                        ui_threshold, ui_batch_size, ui_max_patches, progress_bar
                    )
                    progress_bar.empty()
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.analysis_result = {
                        'overlay': overlay, 'heatmap': heatmap, 'stats': stats,
                        'filename': current_img_name, 'timestamp': timestamp
                    }
                    st.session_state.history.insert(0, {
                        "Time": datetime.datetime.now().strftime("%H:%M"),
                        "File": current_img_name,
                        "Risk": f"{stats['cancer_percentage']}%"
                    })
            except Exception as e:
                st.error("Lỗi hệ thống."); st.code(traceback.format_exc())

        result = st.session_state.analysis_result
        if result and result.get('filename') == current_img_name:
            overlay, heatmap, stats, timestamp = result['overlay'], result['heatmap'], result['stats'], result['timestamp']

            if overlay is None:
                st.warning("⚠️ Không tìm thấy mô tế bào.")
            else:
                st.info(f"📄 Kết quả cho: **{result['filename']}**")
                
                # --- PHẦN KÍNH LÚP (ZOOM) ---
                tab1, tab2 = st.tabs(["🔍 Vùng tổn thương (Zoom)", "🌡️ Bản đồ nhiệt (Zoom)"])
                
                # Chuẩn bị ảnh cho zoom
                heatmap_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                heatmap_color = cv2.cvtColor(cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                blend = cv2.addWeighted(image_array, 0.6, heatmap_color, 0.4, 0)
                
                with tab1:
                    st.caption("👉 Di chuột vào ảnh để soi kính lúp:")
                    image_zoom(Image.fromarray(overlay), mode="mousemove", size=700, zoom_factor=3, keep_aspect_ratio=True)

                with tab2:
                    st.caption("👉 Di chuột vào ảnh để soi kính lúp:")
                    image_zoom(Image.fromarray(blend), mode="mousemove", size=700, zoom_factor=3, keep_aspect_ratio=True)

                # Metrics
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tổng Patch", f"{stats['total_patches']}")
                c2.metric("Số Patch IDC", stats['cancer_patches'])
                color = "inverse" if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT else "normal"
                c3.metric("Tỷ lệ bệnh", f"{stats['cancer_percentage']}%", delta_color=color)
                c4.metric("Max Conf", f"{stats['max_confidence']}")

                if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT:
                    st.error(f"🚨 NGUY CƠ CAO ({stats['cancer_percentage']}%)")
                else:
                    st.success("✅ AN TOÀN")
                
                # Lưu & Tải
                results_dir = config.BASE_DIR / "results"
                path_overlay = results_dir / f"overlay_{timestamp}.png"
                path_json = results_dir / f"stats_{timestamp}.json"
                path_csv = results_dir / f"stats_{timestamp}.csv"

                if not path_csv.exists():
                     try:
                        results_dir.mkdir(exist_ok=True)
                        cv2.imwrite(str(path_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        
                        with open(path_json, "w", encoding="utf-8") as f: json.dump(stats, f, indent=2)
                        
                        stats_csv = stats.copy()
                        stats_csv.update({'timestamp': timestamp, 'image_name': current_img_name, 
                                          'device': config.DEVICE, 'threshold': ui_threshold, 
                                          'batch_size': ui_batch_size, 'max_patches_limit': ui_max_patches})
                        pd.DataFrame([stats_csv]).to_csv(path_csv, index=False)
                     except: pass
                
                # Nút tải
                st.write("---")
                d1, d2 = st.columns(2)
                with d1:
                    with open(path_overlay, "rb") as f: st.download_button("⬇️ Tải Ảnh Kết quả", f, path_overlay.name, "image/png", use_container_width=True)
                with d2:
                    with open(path_json, "rb") as f: st.download_button("⬇️ Tải Số liệu JSON", f, path_json.name, "application/json", use_container_width=True)

                if st.button("🔄 Phân tích ca mới (Reset)", type="secondary", use_container_width=True):
                    st.session_state.analysis_result = None
                    st.rerun()

if __name__ == "__main__":
    main()