import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QScrollArea, QFrame, QTextBrowser,
                             QComboBox, QSlider, QGridLayout, QGroupBox, QFormLayout)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import os

class Section34Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = "anh-trang-den-1.webp"
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)
        
        title = QLabel("Mục 3.4: Hàm Suy Biến (Degradation Function)")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # --- LÝ THUYẾT CHI TIẾT THEO SLIDE ---
        theory_box = QTextBrowser()
        theory_box.setHtml("""
            <div style="font-size: 14px; font-family: Segoe UI, sans-serif; line-height: 1.5;">
                <h3 style="color: #2980b9;">1. Impulse Response & Superposition Integral</h3>
                <p>Ảnh bị suy biến được tính bằng tích phân chập (Superposition/Fredholm integral của loại 1):<br>
                <b>g(x,y) = ∫∫ f(α,β) h(x-α, y-β) dα dβ + η(x,y)</b></p>
                <p>Trong đó <b>h(x,y)</b> là đáp ứng xung (Impulse response) của hàm suy biến. Trong miền tần số: <b>G(u,v) = H(u,v)F(u,v) + N(u,v)</b></p>
                
                <h3 style="color: #2980b9;">2. Các phương pháp ước lượng hàm suy biến (Estimating the degradation function)</h3>
                <ul>
                    <li><b>Bằng quan sát (Observation):</b> $\hat{H}_s(u,v) = G_s(u,v) / \hat{F}_s(u,v)$</li>
                    <li><b>Bằng thực nghiệm (Experimentation):</b> Dùng một xung ánh sáng A để tìm H. $\hat{H}_s(u,v) = G_s(u,v) / A$</li>
                    <li><b>Bằng mô hình toán học (Modeling):</b> Ví dụ mô hình nhiễu động khí quyển (Atmospheric Turbulence):<br>
                    <b>$H_s(u,v) = e^{-k(u^2+v^2)^{5/6}}$</b> (với k là hằng số quyết định mức độ nhiễu động)</li>
                </ul>
            </div>
        """)
        theory_box.setMaximumHeight(300)
        theory_box.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px;")
        layout.addWidget(theory_box)
        
        # --- CONTROL PANEL ---
        control_group = QGroupBox("Cài Đặt Tham Số Mô Phỏng")
        control_group.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        control_layout = QFormLayout(control_group)
        
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "1. Mô hình nhiễu động khí quyển (Atmospheric Turbulence) - Tần số", 
            "2. Mô hình mờ chuyển động (Motion Blur) - Không gian"
        ])
        self.combo_model.currentIndexChanged.connect(self.update_controls)
        control_layout.addRow("Chọn Mô Hình Ước Lượng:", self.combo_model)
        
        # Tham số cho Turbulence
        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_k.setRange(1, 100) # Hệ số scale cho k
        self.slider_k.setValue(25)
        self.lbl_k_val = QLabel("k = 0.0025")
        
        # Tham số cho Motion Blur
        self.slider_length = QSlider(Qt.Orientation.Horizontal)
        self.slider_length.setRange(1, 100)
        self.slider_length.setValue(30)
        self.lbl_length_val = QLabel("Chiều dài = 30")
        
        self.slider_angle = QSlider(Qt.Orientation.Horizontal)
        self.slider_angle.setRange(0, 180)
        self.slider_angle.setValue(45)
        self.lbl_angle_val = QLabel("Góc = 45°")
        
        # Container cho thanh trượt
        box_k = QHBoxLayout()
        box_k.addWidget(self.slider_k)
        box_k.addWidget(self.lbl_k_val)
        self.widget_k = QWidget()
        self.widget_k.setLayout(box_k)
        control_layout.addRow("Hằng số nhiễu động (k):", self.widget_k)
        
        box_l = QHBoxLayout()
        box_l.addWidget(self.slider_length)
        box_l.addWidget(self.lbl_length_val)
        self.widget_l = QWidget()
        self.widget_l.setLayout(box_l)
        control_layout.addRow("Độ dài chuyển động (pixels):", self.widget_l)
        
        box_a = QHBoxLayout()
        box_a.addWidget(self.slider_angle)
        box_a.addWidget(self.lbl_angle_val)
        self.widget_a = QWidget()
        self.widget_a.setLayout(box_a)
        control_layout.addRow("Góc chuyển động:", self.widget_a)
        
        self.slider_k.valueChanged.connect(self.on_k_change)
        self.slider_length.valueChanged.connect(self.on_l_change)
        self.slider_angle.valueChanged.connect(self.on_a_change)
        
        layout.addWidget(control_group)
        
        # --- NÚT CHẠY ---
        self.btn_run = QPushButton("Tính Toán Hàm Suy Biến & Chạy Mô Phỏng")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white; font-weight: bold; font-size: 16px;
                padding: 12px; border-radius: 6px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self.run_simulation)
        layout.addWidget(self.btn_run)

        # --- IMAGES GRID ---
        grid = QGridLayout()
        
        self.lbl_orig = self.create_image_label()
        self.lbl_impulse = self.create_image_label()
        self.lbl_degraded = self.create_image_label()
        self.lbl_response = self.create_image_label()
        
        grid.addWidget(QLabel("<b>Ảnh Gốc f(x,y)</b>"), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_orig, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        
        grid.addWidget(QLabel("<b>Ảnh Suy Biến (Đầu Ra) g(x,y)</b>"), 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_degraded, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        
        grid.addWidget(QLabel("<b>Ảnh Xung Đầu Vào δ(x,y)</b><br/>(Dùng để test đáp ứng)"), 2, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_impulse, 3, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        
        grid.addWidget(QLabel("<b>Đáp Ứng Xung h(x,y)</b><br/>(Kernel/Hình dạng làm mờ phóng to)"), 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_response, 3, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(grid)
        
        # --- CODE SNIPPET ---
        self.code_view = QTextBrowser()
        self.code_view.setStyleSheet("background-color: #282c34; color: #abb2bf; font-family: Consolas, Courier; font-size: 13px; padding: 15px; border-radius: 8px;")
        self.code_view.setMaximumHeight(350)
        layout.addWidget(QLabel("<b>Code Python Xử Lý (Cập nhật theo mô hình):</b>"))
        layout.addWidget(self.code_view)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        self.update_controls()
        self.load_initial_images()

    def create_image_label(self):
        lbl = QLabel("Chưa mô phỏng")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("border: 2px solid #bdc3c7; background-color: #ecf0f1; border-radius: 4px;")
        lbl.setFixedSize(320, 320)
        return lbl

    def update_controls(self):
        idx = self.combo_model.currentIndex()
        if idx == 0: # Turbulence
            self.widget_k.setVisible(True)
            self.widget_l.setVisible(False)
            self.widget_a.setVisible(False)
        else: # Motion
            self.widget_k.setVisible(False)
            self.widget_l.setVisible(True)
            self.widget_a.setVisible(True)

    def on_k_change(self, val):
        self.lbl_k_val.setText(f"k = {val * 0.0001:.4f}")
    def on_l_change(self, val):
        self.lbl_length_val.setText(f"Chiều dài = {val} pixels")
    def on_a_change(self, val):
        self.lbl_angle_val.setText(f"Góc = {val}°")

    def load_initial_images(self):
        try:
            from PIL import Image
            pil_img = Image.open(self.image_path).convert('L')
            self.orig_img = np.array(pil_img)
            # Thay đổi kích thước một chút nếu ảnh quá to để FFT chạy mượt
            if self.orig_img.shape[0] > 800 or self.orig_img.shape[1] > 800:
                self.orig_img = cv2.resize(self.orig_img, (600, 600))
                
            self.display_image(self.orig_img, self.lbl_orig)
            
            # Khởi tạo ảnh xung mô phỏng điểm sáng nhỏ xíu (Impulse)
            vis_impulse = np.zeros((101, 101), dtype=np.uint8)
            vis_impulse[50, 50] = 255
            self.display_image(vis_impulse, self.lbl_impulse, zoom=True)
            
        except Exception as e:
            self.lbl_orig.setText(f"Lỗi đọc ảnh:\n{e}")

    def run_simulation(self):
        if not hasattr(self, 'orig_img'): 
            return
        
        idx = self.combo_model.currentIndex()
        img_float = np.float32(self.orig_img)
        
        if idx == 0: # Mô hình nhiễu động khí quyển (Modeling)
            k = self.slider_k.value() * 0.0001
            
            # Bước 1: Biến đổi Fourier 2D của ảnh
            F = np.fft.fft2(img_float)
            F_shifted = np.fft.fftshift(F) # Đưa tần số 0 ra giữa
            
            # Bước 2: Tạo lưới tần số (u, v)
            rows, cols = img_float.shape
            u = np.arange(-cols/2, cols/2)
            v = np.arange(-rows/2, rows/2)
            U, V = np.meshgrid(u, v)
            
            # Bước 3: Tính toán Hàm suy biến H(u,v) theo công thức toán học
            D2 = U**2 + V**2
            H = np.exp(-k * (D2**(5/6)))
            
            # Bước 4: Áp dụng suy biến G(u,v) = H(u,v) * F(u,v)
            G_shifted = H * F_shifted
            G = np.fft.ifftshift(G_shifted)
            
            # Bước 5: Biến đổi ngược IFFT để ra ảnh g(x,y)
            g = np.abs(np.fft.ifft2(G))
            g = np.clip(g, 0, 255).astype(np.uint8)
            
            # Trích xuất Đáp ứng xung h(x,y) bằng cách làm IFFT trực tiếp của H(u,v)
            H_spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(H)))
            H_spatial = np.fft.fftshift(H_spatial)
            H_vis = cv2.normalize(H_spatial, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Cắt phần tâm của đáp ứng xung để phóng to hiển thị
            cy, cx = rows//2, cols//2
            c_size = 30
            H_vis_cropped = H_vis[cy-c_size:cy+c_size, cx-c_size:cx+c_size]
            
            self.display_image(g, self.lbl_degraded)
            self.display_image(H_vis_cropped, self.lbl_response, zoom=True)
            
            # Cập nhật code view
            code_text = f'''# Ước Lượng Hàm Suy Biến Bằng Mô Hình (Atmospheric Turbulence)
# Công thức: H(u,v) = e^(-k * (u^2 + v^2)^(5/6))
import numpy as np

k = {k:.4f}
# 1. Chuyển ảnh f(x,y) sang miền tần số F(u,v)
F = np.fft.fftshift(np.fft.fft2(img))

# 2. Tạo ma trận hàm suy biến H(u,v)
rows, cols = img.shape
u = np.arange(-cols/2, cols/2)
v = np.arange(-rows/2, rows/2)
U, V = np.meshgrid(u, v)
H = np.exp(-k * ((U**2 + V**2)**(5/6)))

# 3. Nhân phổ (Superposition in frequency): G(u,v) = H(u,v) * F(u,v)
G = H * F

# 4. Chuyển kết quả về miền không gian g(x,y)
g = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
'''
            self.code_view.setPlainText(code_text)

        else: # Mô hình mờ chuyển động (Experimentation / Kernel-based)
            length = self.slider_length.value()
            angle = self.slider_angle.value()
            
            # Tạo Motion Kernel (h(x,y))
            kernel = np.zeros((length, length), dtype=np.float32)
            kernel[int(length/2), :] = np.ones(length, dtype=np.float32)
            M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (length, length))
            kernel = kernel / np.sum(kernel) # Chuẩn hóa năng lượng
            
            # Tích phân chập g(x,y) = f(x,y) * h(x,y)
            g = cv2.filter2D(self.orig_img, -1, kernel)
            
            # Hiển thị đáp ứng xung (kernel) phóng to
            H_vis = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            self.display_image(g, self.lbl_degraded)
            self.display_image(H_vis, self.lbl_response, zoom=True)
            
            code_text = f'''# Mô phỏng Motion Blur trong Miền Không Gian
import cv2
import numpy as np

length = {length}
angle = {angle}

# 1. Tạo Ma trận Đáp Ứng Xung h(x,y) (Motion Blur Kernel)
kernel = np.zeros((length, length), dtype=np.float32)
# Khởi tạo một đường thẳng sáng ở giữa
kernel[int(length/2), :] = np.ones(length, dtype=np.float32)

# Xoay đường thẳng theo góc đã chỉ định
M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1)
kernel = cv2.warpAffine(kernel, M, (length, length))
kernel = kernel / np.sum(kernel) # Đảm bảo bảo toàn năng lượng sáng

# 2. Tích phân chập Superposition: g(x,y) = h(x,y) * f(x,y)
g = cv2.filter2D(img, -1, kernel)
'''
            self.code_view.setPlainText(code_text)

    def display_image(self, img_array, label, zoom=False):
        h, w = img_array.shape
        contiguous_img = np.ascontiguousarray(img_array)
        qimg = QImage(contiguous_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit fixed size label
        label_w, label_h = label.width(), label.height()
        if zoom:
            # Phóng to nội suy mờ mờ cho các ảnh nhỏ (như kernel / impulse)
            pixmap = pixmap.scaled(label_w, label_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        else:
            pixmap = pixmap.scaled(label_w, label_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
        label.setPixmap(pixmap)
