# idc-classification-thesis
Deep learning application (CNN-DeiT) builds a breast cancer classification model from IDC histopathology images.
# Đường dẫn file
file_path = "/content/drive/MyDrive/BreastCancer_project/Webapp/README.md"

# Nội dung README
readme_content = """# Ứng dụng học sâu (Hybrid CNN-DeiT) xây dựng mô hình phân loại ung thư vú từ ảnh mô bệnh học IDC

**Đề án Thạc sĩ Khoa học Máy tính**
**Tác giả:** Vũ Hữu Hoàng
**Khóa:** 24.1
**Trường:** Đại học Sài Gòn

---

## 1. Giới thiệu
Đây là ứng dụng Web demo cho đề án *"Ứng dụng học sâu (Hybrid CNN-DeiT) xây dựng mô hình phân loại ung thư vú từ ảnh mô bệnh học IDC"*.
Hệ thống sử dụng mô hình lai **Hybrid CNN-DeiT** kết hợp với kỹ thuật làm mịn nhãn **EMA** để phát hiện các vùng nghi ngờ ung thư trên ảnh vi thể mô bệnh học (Whole Slide Image patches).

### Tính năng chính:
- Tải lên ảnh mô bệnh học (H&E).
- Tự động phát hiện và khoanh vùng ung thư xâm lấn (IDC).
- Hiển thị bản đồ nhiệt (Heatmap) độ tin cậy của mô hình.
- Báo cáo thống kê định lượng tỷ lệ vùng bệnh.

---

## 2. Yêu cầu hệ thống
- **Hệ điều hành:** Windows 10/11, Linux (Ubuntu), hoặc macOS.
- **Python:** Phiên bản 3.10 trở lên.
- **RAM:** Tối thiểu 8GB (Khuyến nghị 16GB nếu chạy trên CPU).
- **GPU (Tùy chọn):** NVIDIA GPU với VRAM >= 4GB (để tốc độ xử lý nhanh hơn).

---

## 3. Cấu trúc thư mục

```text
Webapp/
├── app.py                 # File khởi chạy ứng dụng
├── config.py              # File cấu hình
├── requirements.txt       # Danh sách thư viện
├── README.md              # Hướng dẫn sử dụng
├── packages/              # (Offline) Thư viện tải sẵn
├── assets/                # Logo, hình ảnh giao diện
├── models/                # File trọng số mô hình
│   └── hybrid_best_ema31.pth
└── src/                   # Mã nguồn lõi
    ├── model_hybrid1.py
```

---

## 4. Hướng dẫn Cài đặt
Cách 1: Máy có Internet
Mở Terminal tại thư mục dự án và chạy:

**pip install -r requirements.txt**

Cách 2: Máy KHÔNG có Internet (Offline)
Yêu cầu: Đã chép thư mục packages (chứa các file .whl) vào dự án. Chạy lệnh sau:

**pip install --no-index --find-links=./packages -r requirements.txt**

---

## 5. Hướng dẫn Sử dụng
Bước 1: Khởi chạy ứng dụng

**streamlit run app.py**

Bước 2: Truy cập Web App
Hệ thống sẽ mở trình duyệt mặc định hoặc truy cập qua:

*Local URL: http://localhost:8501*

Bước 3: Thao tác
Sidebar: Điều chỉnh "Ngưỡng tin cậy" (Confidence Threshold), chọn "Sử dụng GPU" nếu có card rời.

Upload ảnh: Nhấn nút "Browse files".

Phân tích: Nhấn nút "PHÂN TÍCH NGAY".

Xem kết quả: Quan sát vùng khoanh đỏ trên ảnh gốc và chuyển sang tab "Bản đồ nhiệt" để xem chi tiết.

---

## 6. Ghi chú kỹ thuật
Mô hình Hybrid CNN-DeiT đã được huấn luyện trên tập dữ liệu chuẩn hóa.

Tối ưu tốc độ trên CPU bằng kỹ thuật Tissue Masking (loại bỏ nền trắng).

Thư viện OpenCV sử dụng phiên bản headless để tương thích tốt nhất với môi trường server/cloud.

---

## 7. Liên hệ
Mọi thắc mắc xin liên hệ tác giả: vuhuuhoang1080@gmail.com

© 2026 Vũ Hữu Hoàng. All Rights Reserved. """
with open(file_path, "w", encoding="utf-8") as f: f.write(readme_content)

print(f"✅ Đã cập nhật nội dung README.md mới tại: {file_path}")
