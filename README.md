# IMP302 - Image Processing Demo Presentation

Đây là dự án mô phỏng các khái niệm cốt lõi trong môn Xử lý Tín hiệu Số (DSP) / Xử lý Ảnh, đặc biệt tập trung vào phần: **Image Degradation - Restoration and Reconstruction** (Chương 3).

Dự án bao gồm 2 phiên bản giao diện:
1. **Phiên bản Web:** Được xây dựng bằng Streamlit, giao diện hiện đại, dễ thao tác và trình bày.
2. **Phiên bản Desktop App:** Được xây dựng bằng PyQt6 dành cho ứng dụng máy tính truyền thống.

---

## Hướng dẫn cài đặt

Để chạy được mã nguồn của dự án này, máy tính của bạn cần cài đặt sẵn Python (phiên bản 3.8 trở lên). Sau đó, bạn chỉ cần mở Terminal (hoặc Command Prompt) tại thư mục chứa dự án và chạy lệnh sau để cài đặt toàn bộ thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

## Hướng dẫn chạy ứng dụng

### 1. Khởi chạy phiên bản Web (Được khuyên dùng cho Thuyết trình)
Phiên bản Web có giao diện cực kỳ đẹp mắt, được thiết kế với các slider tương tác trực tiếp theo thời gian thực để minh họa các định lý Toán học một cách sinh động nhất (đặc biệt là mục 3.4 Hàm Suy Biến).

Để khởi chạy, bạn nhập lệnh sau vào Terminal:
```bash
streamlit run code_demo_presentation/app_web.py
```
*(Ngay sau khi chạy lệnh, trình duyệt web mặc định của bạn sẽ tự động mở lên ở địa chỉ `http://localhost:8501`)*

### 2. Khởi chạy phiên bản Desktop App
Nếu bạn muốn dùng thử phiên bản ứng dụng giao diện cửa sổ truyền thống (PyQt6), hãy nhập lệnh sau:
```bash
python code_demo_presentation/main.py
```

---

## Cấu trúc thư mục
- `code_demo_presentation/`: Thư mục chứa toàn bộ mã nguồn của phần mềm mô phỏng.
  - `app_web.py`: File khởi chạy giao diện Web.
  - `main.py`: File khởi chạy giao diện Desktop.
  - `section_3_4.py`: Các component xử lý giao diện dành riêng cho Desktop App.
- `Student/`: Thư mục chứa các file PDF tài liệu tham khảo cho môn học.
- `requirements.txt`: Chứa danh sách các thư viện Python phụ thuộc.
