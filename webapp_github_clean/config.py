import os
import torch
from pathlib import Path

# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (PATHS)
# ============================================================
# Lấy đường dẫn thư mục hiện tại của file config.py (Webapp/)
BASE_DIR = Path(__file__).parent.absolute()

# Các thư mục con
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
SRC_DIR = BASE_DIR / "src"

# Đường dẫn file trọng số mô hình (Checkpoint)
# Đảm bảo tên file khớp với file bạn đã upload vào thư mục models/
MODEL_PATH = MODELS_DIR / "hybrid_best_ema31.pth"

# Đường dẫn Logo trường (Nếu chưa có thì để None hoặc đường dẫn ảnh mặc định)
LOGO_PATH = ASSETS_DIR / "logo_DHSG.png" if (ASSETS_DIR / "logo_DHSG.png").exists() else None

# Đường dẫn ảnh mẫu (Sample Images) để demo nhanh
SAMPLE_DIR = ASSETS_DIR / "samples"
SAMPLE_IMAGES = {
    "Ca Ung thư (IDC)": SAMPLE_DIR / "sample_idc.png",
    "Ca Lành tính": SAMPLE_DIR / "sample_benign.png",
    "Ca Nhiễu/Mô đệm": SAMPLE_DIR / "sample_stroma.png"
}

# Đường dẫn ảnh minh họa kiến trúc (nếu có)
MODEL_VIZ_PATH = ASSETS_DIR / "architecture.png"

# ============================================================
# 2. CẤU HÌNH MÔ HÌNH (MODEL HYPERPARAMETERS)
# ============================================================
# Các tham số này BẮT BUỘC phải khớp với lúc huấn luyện (Training)
MODEL_PARAMS = {
    "num_classes": 2,
    "img_size": (224, 224),
    "embed_dim": 448,   # Khớp với file inference_wsi5.py
    "depth": 8,
    "num_heads": 8,
    "mlp_ratio": 4.0,
}

# Chuẩn hóa ảnh (ImageNet stats)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Nhãn lớp
CLASS_NAMES = {
    0: "Non-IDC (Lành tính/Bình thường)",
    1: "IDC (Ung thư xâm lấn)"
}

# ============================================================
# 3. CẤU HÌNH SUY LUẬN (INFERENCE SETTINGS)
# ============================================================
# Kích thước cắt patch từ ảnh gốc (WSI)
PATCH_SIZE = 50 

# Bước nhảy khi cắt ảnh (Stride). 
# Nếu bằng PATCH_SIZE thì cắt không chồng lấn.
# Nếu nhỏ hơn PATCH_SIZE thì cắt chồng lấn (kỹ hơn nhưng chậm hơn).
STRIDE = 50 

# Thiết bị xử lý (Tự động phát hiện)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch size: GPU thì để cao (128-256), CPU thì để thấp (32) để tránh treo máy
BATCH_SIZE = 128 if DEVICE == "cuda" else 32

# Số luồng CPU tải dữ liệu
NUM_WORKERS = 0

# ============================================================
# 4. CẤU HÌNH GIAO DIỆN (UI SETTINGS)
# ============================================================
APP_TITLE = "HỆ THỐNG HỖ TRỢ PHÂN LOẠI UNG THƯ VÚ (IDC)"
APP_ICON = "🧬"
APP_DESCRIPTION = """ 
**Mô hình:** Hybrid CNN-DeiT (EMA) 
**Phiên bản:** v1.0 
**Tác giả:** Vũ Hữu Hoàng - CH11241003 
**Đề Thạc sĩ Khoa học Máy tính - K24.1** 
""" 
CONFIDENCE_THRESHOLD = 0.5 
RESULTS_DIR = BASE_DIR / "results" 
RESULTS_DIR.mkdir(exist_ok=True)

# Ngưỡng cảnh báo (Nếu tỷ lệ ung thư vượt quá số này sẽ hiện cảnh báo đỏ)
DANGER_THRESHOLD_PERCENT = 20.0