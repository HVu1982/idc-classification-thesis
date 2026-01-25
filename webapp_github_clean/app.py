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
import gdown  # Thư viện để tải file từ Drive

# --- 1. SETUP & IMPORT CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import config
except ImportError:
    st.error("❌ Không tìm thấy file config.py. Vui lòng kiểm tra lại thư mục.")
    st.stop()

sys.path.append(str(config.SRC_DIR))

# --- 2. CẤU HÌNH TRANG ---
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
# 📥 TỰ ĐỘNG TẢI MODEL TỪ GOOGLE DRIVE
# ============================================================
# ⚠️ THAY ID CỦA BẠN VÀO DƯỚI ĐÂY (ID lấy từ link share file .pth)
MODEL_DRIVE_ID = "1Ruvjg57t-JLoP1QcWK_I8UzcFuUFjCnN" 

@st.cache_resource
def download_model_from_drive():
    """Tải model từ Google Drive nếu chưa tồn tại"""
    if not config.MODEL_PATH.exists():
        # Tạo thư mục models nếu chưa có
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
        output = str(config.MODEL_PATH)
        
        st.toast("⏳ Đang tải Model từ Cloud (Lần đầu chạy mất ~1 phút)...", icon="cloud")
        try:
            gdown.download(url, output, quiet=False)
            st.success("✅ Tải Model thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi tải model: {e}")
            st.stop()

# ============================================================
# 3. CLASS & CORE FUNCTIONS
# ============================================================

class WSIPatchDataset(Dataset):
    def __init__(self, image, coords, patch_size=50, transform=None):
        self.image = image
        self.coords = coords
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch = self.image[y : y + self.patch_size, x : x + self.patch_size]
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        if self.transform:
            patch = self.transform(patch)
        return patch

@st.cache_resource
def load_model(device_name):
    """Load model"""
    # Đảm bảo model đã được tải về trước khi load
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
    
    if not coords:
        return None, None, {"cancer_percentage": 0.0}

    total_found = len(coords)
    if max_patches > 0 and total_found > max_patches:
        coords = coords[:max_patches]
        st.toast(f"⚡ Giới hạn xử lý: {max_patches}/{total_found} patches", icon="🚀")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(config.MODEL_PARAMS['img_size']),
        T.ToTensor(),
        T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    dataset = WSIPatchDataset(image_array, coords, patch_size, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predictions.extend((probs >= threshold).int().cpu().numpy())
            confidences.extend(probs.cpu().numpy())
            if progress_bar:
                prog_val = (i + 1) / len(loader)
                progress_bar.progress(prog_val, text=f"Đang xử lý batch {i+1}/{len(loader)}...")

    heatmap = np.zeros((h, w), dtype=np.float32)
    overlay = image_array.copy()
    cancer_count = 0
    
    for (y, x), pred, conf in zip(coords, predictions, confidences):
        heatmap[y : y + patch_size, x : x + patch_size] = conf
        if pred == 1:
            cancer_count += 1
            cv2.rectangle(overlay, (x, y), (x + patch_size, y + patch_size), (255, 0, 0), 2)
            
    stats = {
        "total_patches": len(coords),
        "original_patches": total_found,
        "cancer_patches": cancer_count,
        "cancer_percentage": round((cancer_count / len(coords)) * 100, 2),
        "max_confidence": round(float(np.max(confidences)), 4) if confidences else 0
    }
    
    return overlay, heatmap, stats

# ============================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (MAIN)
# ============================================================
def main():
    # Gọi hàm tải model ngay khi app khởi động
    download_model_from_drive()

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    # === SIDEBAR ===
    with st.sidebar:
        if config.LOGO_PATH and config.LOGO_PATH.exists():
            st.image(str(config.LOGO_PATH), width=120)
        
        st.header("⚙️ Cấu hình hệ thống")
        # Ép hiển thị CPU nếu chạy trên Cloud (thường Cloud free không có GPU)
        device_display = "CPU (Cloud)" if not torch.cuda.is_available() else "GPU (CUDA)"
        st.info(f"Thiết bị: **{device_display}**")

        with st.expander("🛠️ Tham số Mô hình", expanded=False):
            st.markdown(f"- Model: Hybrid CNN-DeiT\n- Patch Size: {config.PATCH_SIZE}px")
            if hasattr(config, 'MODEL_VIZ_PATH') and config.MODEL_VIZ_PATH.exists():
                st.image(str(config.MODEL_VIZ_PATH), caption="Kiến trúc đề xuất", use_column_width=True)
            
            ui_max_patches = st.slider("⚡ Giới hạn Patch (Demo)", 0, 5000, 0, 100)

        ui_threshold = st.slider("Ngưỡng (Threshold)", 0.0, 1.0, config.CONFIDENCE_THRESHOLD, 0.05)
        
        default_bs_idx = 0 # Cloud nên để batch nhỏ (16) để tránh tràn RAM
        ui_batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=default_bs_idx)

        if st.session_state.history:
            st.markdown("---")
            st.subheader("🕒 Lịch sử phiên này")
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

        st.caption("© 2026 Vũ Hữu Hoàng")

    # === MAIN CONTENT ===
    st.title(config.APP_TITLE)
    st.write("---")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("1. Tải ảnh đầu vào")
        input_source = st.radio("Nguồn ảnh:", ["Tải ảnh lên", "Dùng ảnh mẫu (Demo)"], horizontal=True)
        uploaded_file = None
        current_img_name = ""

        if input_source == "Tải ảnh lên":
            uploaded_file = st.file_uploader("Chọn ảnh H&E (JPG, PNG)", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                current_img_name = uploaded_file.name
                image_pil = Image.open(uploaded_file).convert('RGB')
        else:
            if hasattr(config, 'SAMPLE_IMAGES'):
                sample_choice = st.selectbox("Chọn ca bệnh mẫu:", list(config.SAMPLE_IMAGES.keys()))
                sample_path = config.SAMPLE_IMAGES[sample_choice]
                if sample_path.exists():
                    image_pil = Image.open(sample_path).convert('RGB')
                    current_img_name = sample_path.name
                    class MockFile: name = current_img_name
                    uploaded_file = MockFile()

        if 'image_pil' in locals() and image_pil:
            image_array = np.array(image_pil)
            st.image(image_pil, caption=f"Ảnh đầu vào: {current_img_name}", use_column_width=True)
            analyze_trigger = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)
        else:
            analyze_trigger = False

    with col2:
        st.subheader("2. Kết quả Chẩn đoán")
        if uploaded_file and analyze_trigger:
            progress_bar = st.progress(0, text="Khởi tạo mô hình...")
            try:
                # Cloud thường không có GPU, ép dùng CPU nếu cần
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
                    st.session_state.history.insert(0, {"Thời gian": datetime.datetime.now().strftime("%H:%M"), "Ảnh": current_img_name, "Tỷ lệ": f"{stats['cancer_percentage']}%"})

            except Exception as e:
                st.error("Lỗi hệ thống."); st.code(traceback.format_exc())

        result = st.session_state.analysis_result
        if result and uploaded_file and result['filename'] == current_img_name:
            # (Phần hiển thị kết quả giữ nguyên như cũ)
            overlay, heatmap, stats, timestamp = result['overlay'], result['heatmap'], result['stats'], result['timestamp']
            
            if overlay is None:
                st.warning("Không tìm thấy mô tế bào.")
            else:
                st.info(f"Kết quả: **{result['filename']}**")
                tab1, tab2 = st.tabs(["Vùng tổn thương", "Bản đồ nhiệt"])
                heatmap_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                heatmap_color = cv2.cvtColor(cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                blend = cv2.addWeighted(image_array, 0.6, heatmap_color, 0.4, 0)

                with tab1: st.image(overlay, caption="Phát hiện IDC", use_column_width=True)
                with tab2: st.image(blend, caption="Heatmap", use_column_width=True)

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tổng Patch", stats['total_patches'])
                c2.metric("IDC Patch", stats['cancer_patches'])
                c3.metric("Tỷ lệ bệnh", f"{stats['cancer_percentage']}%")
                c4.metric("Max Conf", stats['max_confidence'])

                if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT:
                    st.error(f"🚨 NGUY CƠ CAO")
                else:
                    st.success("✅ AN TOÀN")

if __name__ == "__main__":
    main()