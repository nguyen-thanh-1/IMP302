import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QStackedWidget, QLabel)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon

# Sau này, chúng ta sẽ import các module xử lý cho từng phần tại đây
# Ví dụ:
# from section_3_5 import Section35Widget
# from section_3_6 import Section36Widget
from section_3_4 import Section34Widget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mô phỏng Xử lý Ảnh - Chương 3")
        self.resize(1200, 800)
        self.setup_ui()

    def setup_ui(self):
        # CSS (StyleSheet) giúp giao diện hiện đại và đẹp hơn
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #dcdfe6;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 15px;
                font-weight: bold;
                color: #606266;
            }
            QPushButton:hover {
                background-color: #ecf5ff;
                color: #409eff;
                border: 1px solid #c6e2ff;
            }
            QPushButton:checked {
                background-color: #409eff;
                color: #ffffff;
                border: none;
            }
            QLabel.title {
                font-size: 28px;
                font-weight: bold;
                color: #303133;
                margin-bottom: 20px;
            }
            QLabel.subtitle {
                font-size: 16px;
                color: #909399;
            }
        """)

        # Widget chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === Thanh công cụ (Navigation Bar) ===
        nav_layout = QHBoxLayout()
        
        self.btn_3_4 = QPushButton("3.4 Hàm Suy Biến (Degradation)")
        self.btn_3_5 = QPushButton("3.5 Khử Nhiễu Wavelet")
        self.btn_3_6 = QPushButton("3.6 Lọc Wiener (Khôi phục)")
        
        self.btn_3_4.setCheckable(True)
        self.btn_3_5.setCheckable(True)
        self.btn_3_6.setCheckable(True)
        
        # Thêm hiệu ứng trỏ chuột
        self.btn_3_4.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_3_5.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_3_6.setCursor(Qt.CursorShape.PointingHandCursor)
        
        nav_layout.addWidget(self.btn_3_4)
        nav_layout.addWidget(self.btn_3_5)
        nav_layout.addWidget(self.btn_3_6)
        nav_layout.addStretch() # Đẩy các nút sang trái

        main_layout.addLayout(nav_layout)

        # === Khu vực nội dung chính (Stacked Widget) ===
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("""
            QStackedWidget {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #ebeef5;
            }
        """)
        main_layout.addWidget(self.stacked_widget)

        # Tạo các trang giữ chỗ tạm thời (Placeholder)
        # Sau này ta sẽ thay thế bằng class thực tế từ các file khác
        self.page_3_4 = Section34Widget()
        self.page_3_5 = self.create_placeholder_page("Mục 3.5: Khử nhiễu Wavelet", "Trang này sẽ mô phỏng DWT, ngưỡng hóa Thresholding (Soft/Hard) và tái tạo ảnh.")
        self.page_3_6 = self.create_placeholder_page("Mục 3.6: Bộ lọc Wiener", "Trang này mô phỏng quá trình khôi phục ảnh từ nhiễu bằng bộ lọc Wiener tối ưu.")

        self.stacked_widget.addWidget(self.page_3_4)
        self.stacked_widget.addWidget(self.page_3_5)
        self.stacked_widget.addWidget(self.page_3_6)

        # Kết nối sự kiện bấm nút
        self.btn_3_4.clicked.connect(lambda: self.switch_page(0, self.btn_3_4))
        self.btn_3_5.clicked.connect(lambda: self.switch_page(1, self.btn_3_5))
        self.btn_3_6.clicked.connect(lambda: self.switch_page(2, self.btn_3_6))

        # Hiển thị mặc định mục đầu tiên
        self.switch_page(0, self.btn_3_4)

    def create_placeholder_page(self, title, description):
        """Tạo một trang tạm thời với tiêu đề và mô tả."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title = QLabel(title)
        lbl_title.setProperty("class", "title")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_desc = QLabel(description)
        lbl_desc.setProperty("class", "subtitle")
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_desc)
        return widget

    def switch_page(self, index, active_button):
        """Chuyển đổi giữa các trang."""
        self.stacked_widget.setCurrentIndex(index)
        
        # Reset trạng thái các nút
        self.btn_3_4.setChecked(False)
        self.btn_3_5.setChecked(False)
        self.btn_3_6.setChecked(False)
        
        # Bật trạng thái nút hiện tại
        active_button.setChecked(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Font chữ đẹp hơn
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
