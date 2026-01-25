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
# 📥 TỰ ĐỘNG TẢI MODEL
# ============================================================
MODEL_DRIVE_ID = "1AbC...XYZ_ID_CUA_BAN"  # <--- Thay ID của bạn vào đây

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
        if not config.MODEL_PATH.exists(): return None
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Lỗi Model: {e}")
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
    
    if not coords: return None, None, {"cancer_percentage": 0.0}

    total_found = len(coords)
    if max_patches > 0 and total_found > max_patches:
        coords = coords[:max_patches]
        st.toast(f"⚡ Giới hạn: {max_patches}/{total_found} patches", icon="🚀")

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
                progress_bar.progress((i + 1) / len(loader), text=f"Batch {i+1}/{len(loader)}")

    heatmap = np.zeros((h, w), dtype=np.float32)
    overlay_layer = image_array.copy()
    cancer_count = 0
    gap = 2
    
    for (y, x), pred, conf in zip(coords, predictions, confidences):
        heatmap[y:y+patch_size, x:x+patch_size] = conf
        if pred == 1:
            cancer_count += 1
            cv2.rectangle(overlay_layer, (x+gap, y+gap), (x+patch_size-gap, y+patch_size-gap), (255, 0, 0), -1)

    overlay = cv2.addWeighted(image_array, 0.6, overlay_layer, 0.4, 0)
            
    stats = {
        "total_patches": len(coords), "original_patches": total_found,
        "cancer_patches": cancer_count,
        "cancer_percentage": round((cancer_count / len(coords)) * 100, 2),
        "max_confidence": round(float(np.max(confidences)), 4) if confidences else 0
    }
    return overlay, heatmap, stats

# ============================================================
# 4. TRANG LỊCH SỬ (HISTORY PAGE) - TÍNH NĂNG MỚI
# ============================================================
def render_history_page():
    st.title("🗂️ Lịch sử Chẩn đoán")
    results_dir = config.BASE_DIR / "results"
    
    if not results_dir.exists():
        st.warning("Chưa có dữ liệu lịch sử.")
        return

    # Quét tất cả file stats_*.json
    json_files = sorted(list(results_dir.glob("stats_*.json")), reverse=True)
    
    if not json_files:
        st.info("Chưa có bản ghi nào được lưu.")
        return

    # Tạo bảng chọn
    history_data = []
    for f in json_files:
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
                # Lấy timestamp từ tên file stats_YYYYMMDD_HHMMSS.json
                ts_str = f.stem.replace("stats_", "")
                # Format lại ngày giờ cho đẹp
                dt_obj = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                display_time = dt_obj.strftime("%d/%m/%Y %H:%M:%S")
                
                history_data.append({
                    "Thời gian": display_time,
                    "ID": ts_str,
                    "Ảnh gốc": f"{f.stem.replace('stats_', '')}", # Tên tạm
                    "Tỷ lệ bệnh": f"{data.get('cancer_percentage', 0)}%",
                    "Số Patch IDC": data.get('cancer_patches', 0)
                })
        except: continue

    df = pd.DataFrame(history_data)
    
    # Hiển thị bảng chọn
    col_list, col_detail = st.columns([1, 2])
    
    with col_list:
        st.subheader("Danh sách ca bệnh")
        selected_id = st.radio("Chọn thời gian:", df["ID"].tolist(), format_func=lambda x: f"Ca {x}")

    # Hiển thị chi tiết khi chọn
    if selected_id:
        with col_detail:
            st.subheader(f"🔍 Chi tiết ca: {selected_id}")
            
            # Tìm lại file ảnh dựa trên ID
            p_overlay = results_dir / f"overlay_{selected_id}.png"
            p_heatmap = results_dir / f"heatmap_{selected_id}.png"
            p_json = results_dir / f"stats_{selected_id}.json"
            
            if p_overlay.exists() and p_heatmap.exists():
                # Load stats
                with open(p_json, 'r') as f: stats = json.load(f)
                
                # Hiển thị Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Tỷ lệ bệnh", f"{stats['cancer_percentage']}%")
                m2.metric("Số Patch IDC", stats['cancer_patches'])
                m3.metric("Max Conf", stats.get('max_confidence', 0))
                
                # Hiển thị Ảnh
                tab1, tab2 = st.tabs(["Vùng tổn thương", "Bản đồ nhiệt"])
                with tab1:
                    st.image(str(p_overlay), caption="Kết quả lưu trữ", use_column_width=True)
                with tab2:
                    st.image(str(p_heatmap), caption="Heatmap lưu trữ", use_column_width=True)
                
                # Nút tải lại
                with open(p_overlay, "rb") as f:
                    st.download_button("⬇️ Tải ảnh này", f, f"overlay_{selected_id}.png", "image/png")
            else:
                st.error("⚠️ Không tìm thấy file ảnh gốc của ca này (có thể đã bị xóa).")

# ============================================================
# 5. MAIN APP
# ============================================================
def main():
    # --- SIDEBAR MENU ---
    with st.sidebar:
        if config.LOGO_PATH and config.LOGO_PATH.exists():
            st.image(str(config.LOGO_PATH), width=100)
        
        st.title("Menu Chính")
        app_mode = st.radio("Chức năng:", ["🚀 Phân tích mới", "🗂️ Xem lại Lịch sử"])
        st.write("---")

    # --- CHẾ ĐỘ 1: PHÂN TÍCH MỚI ---
    if app_mode == "🚀 Phân tích mới":
        st.title(config.APP_TITLE)
        
        # Sidebar Config (Chỉ hiện khi phân tích)
        with st.sidebar:
            st.header("⚙️ Cấu hình")
            dev_show = "GPU" if torch.cuda.is_available() else "CPU"
            st.info(f"Thiết bị: **{dev_show}**")
            ui_threshold = st.slider("Ngưỡng (Threshold)", 0.0, 1.0, config.CONFIDENCE_THRESHOLD, 0.05)
            ui_max_patches = st.slider("Giới hạn Patch (Demo)", 0, 5000, 0, 100)
            ui_batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=3 if config.DEVICE=="cuda" else 1)
            st.caption("© 2026 Vũ Hữu Hoàng")

        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("1. Chọn dữ liệu")
            input_source = st.radio("Nguồn ảnh:", ["Tải ảnh lên", "Dùng ảnh mẫu"], horizontal=True)
            image_pil = None
            current_img_name = ""

            if input_source == "Tải ảnh lên":
                uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
                if uploaded_file:
                    image_pil = Image.open(uploaded_file).convert('RGB')
                    current_img_name = uploaded_file.name
            else:
                if hasattr(config, 'SAMPLE_IMAGES'):
                    sample_choice = st.selectbox("Chọn mẫu:", list(config.SAMPLE_IMAGES.keys()))
                    sample_path = config.SAMPLE_IMAGES[sample_choice]
                    if sample_path.exists():
                        image_pil = Image.open(sample_path).convert('RGB')
                        current_img_name = sample_path.name

            if image_pil:
                st.image(image_pil, caption=f"Input: {current_img_name}", use_column_width=True)
                analyze_btn = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)
            else:
                analyze_btn = False

        with col2:
            st.subheader("2. Kết quả")
            if analyze_btn and image_pil:
                image_array = np.array(image_pil)
                progress_bar = st.progress(0, text="Khởi tạo...")
                
                try:
                    run_device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = load_model(run_device)
                    if model:
                        overlay, heatmap, stats = run_inference(
                            model, image_array, run_device, 
                            ui_threshold, ui_batch_size, ui_max_patches, progress_bar
                        )
                        progress_bar.empty()
                        
                        # --- HIỂN THỊ ---
                        t1, t2 = st.tabs(["Vùng bệnh (Zoom)", "Heatmap (Zoom)"])
                        
                        # Xử lý màu heatmap
                        hm_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                        hm_color = cv2.cvtColor(cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                        blend = cv2.addWeighted(image_array, 0.6, hm_color, 0.4, 0)

                        with t1:
                            st.caption("👉 Di chuột để soi chi tiết:")
                            image_zoom(Image.fromarray(overlay), mode="mousemove", size=700, zoom_factor=3)
                        with t2:
                            image_zoom(Image.fromarray(blend), mode="mousemove", size=700, zoom_factor=3)

                        # Metrics
                        st.divider()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Tổng Patch", stats['total_patches'])
                        c2.metric("Số Patch IDC", stats['cancer_patches'])
                        
                        color = "inverse" if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT else "normal"
                        c3.metric("Tỷ lệ bệnh", f"{stats['cancer_percentage']}%", delta_color=color)

                        if stats['cancer_percentage'] >= config.DANGER_THRESHOLD_PERCENT:
                            st.error(f"🚨 NGUY CƠ CAO ({stats['cancer_percentage']}%)")
                        else:
                            st.success("✅ AN TOÀN")

                        # --- TỰ ĐỘNG LƯU ---
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        res_dir = config.BASE_DIR / "results"
                        res_dir.mkdir(exist_ok=True)
                        
                        # Lưu file
                        p_over = res_dir / f"overlay_{ts}.png"
                        p_heat = res_dir / f"heatmap_{ts}.png"
                        p_json = res_dir / f"stats_{ts}.json"
                        p_csv  = res_dir / f"stats_{ts}.csv"
                        
                        try:
                            cv2.imwrite(str(p_over), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(p_heat), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
                            with open(p_json, "w", encoding="utf-8") as f: json.dump(stats, f, indent=2)
                            
                            s_csv = stats.copy()
                            s_csv.update({'timestamp': ts, 'image_name': current_img_name})
                            pd.DataFrame([s_csv]).to_csv(p_csv, index=False)
                            
                            st.success(f"💾 Đã lưu kết quả vào Lịch sử (Mã: {ts})")
                        except: pass

                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # --- CHẾ ĐỘ 2: XEM LỊCH SỬ ---
    elif app_mode == "🗂️ Xem lại Lịch sử":
        render_history_page()

if __name__ == "__main__":
    main()