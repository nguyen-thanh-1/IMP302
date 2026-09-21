import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Mô phỏng Xử lý Ảnh - Chương 3",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS tùy chỉnh Neumorphism ---
st.markdown("""
<style>
    /* Ẩn sidebar và header mặc định của Streamlit */
    [data-testid="collapsedControl"] { display: none; }
    header { display: none !important; }
    
    /* Làm to và đậm chữ ở các Tab */
    button[data-baseweb="tab"] > div > div > p {
        font-size: 20px !important;
        font-weight: 800 !important;
    }
    .stTabs button[role="tab"] p {
        font-size: 20px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stTabs"] button p {
        font-size: 20px !important;
        font-weight: 800 !important;
    }
    
    /* Ép nền sáng cho toàn bộ ứng dụng */
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #e0e5ec !important;
    }
    
    /* ÉP LÊN TRÊN CHÚT NỮA NHƯNG KHÔNG ĐỂ BỊ CẮT */
    .block-container {
        padding-top: 0.8rem !important; /* Đẩy lên vừa khít mép trên */
        margin-top: 0px !important; 
    }
    
    /* Bỏ khoảng trống quanh các cột của Streamlit */
    [data-testid="column"] {
        padding: 0 !important;
    }
    div.stButton {
        padding: 5px 0px; 
    }

    /* Style cho nút bấm Neumorphism */
    .stButton > button {
        width: 100%;
        border-radius: 50px !important; 
        padding: 10px 0px !important; 
        background-color: #e0e5ec !important;
        border: none !important;
        outline: none !important;
        
        box-shadow: 
            4px 4px 8px rgba(163,177,198,0.6), 
            -4px -4px 8px rgba(255,255,255, 0.9) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Chữ bên trong nút */
    .stButton > button p {
        margin: 0 !important;
        font-size: 14px !important; 
        font-weight: 700 !important; 
        color: #111111 !important; 
    }
    
    /* Hiệu ứng Hover */
    .stButton > button:hover {
        transform: translateY(1px);
        box-shadow: 
            2px 2px 5px rgba(163,177,198,0.6), 
            -2px -2px 5px rgba(255,255,255, 0.9) !important;
    }
    .stButton > button:hover p {
        color: #7B61FF !important; 
    }
    
    /* Hiệu ứng Active / Đang chọn (Giữ màu tím) */
    .stButton > button:active, 
    .stButton > button[kind="primary"] {
        background-color: #e0e5ec !important;
        box-shadow: 
            inset 4px 4px 8px rgba(163,177,198, 0.7),
            inset -4px -4px 8px rgba(255,255,255, 0.9) !important;
        transform: translateY(1px);
    }
    .stButton > button:active p,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] div {
        color: #7B61FF !important; 
    }
    
    /* Xóa margin mặc định của Markdown chứa cái vạch */
    .divider-container {
        margin-top: -30px !important; /* Hút sát vạch lên để thu hẹp taskbar */
        margin-bottom: 0px !important;
    }

    /* Đường kẻ chìm ngăn cách */
    .neumorphic-divider {
        border: none;
        height: 2px;
        border-radius: 1px;
        background: #e0e5ec;
        box-shadow: inset 1px 1px 3px rgba(163, 177, 198, 0.6), inset -1px -1px 3px rgba(255, 255, 255, 0.9);
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Quản lý trạng thái trang hiện tại ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "3.4 Hàm Suy Biến"

# --- THANH TASKBAR ---
spacer_left, col1, col2, col3, spacer_right = st.columns([3, 1.2, 1.2, 1.2, 3], gap="small")

with col1:
    if st.button("3.4 Hàm Suy Biến", use_container_width=True, type="primary" if st.session_state.current_page == "3.4 Hàm Suy Biến" else "secondary"):
        st.session_state.current_page = "3.4 Hàm Suy Biến"
        st.rerun()

with col2:
    if st.button("3.5 Khử Nhiễu Wavelet", use_container_width=True, type="primary" if st.session_state.current_page == "3.5 Khử Nhiễu Wavelet" else "secondary"):
        st.session_state.current_page = "3.5 Khử Nhiễu Wavelet"
        st.rerun()

with col3:
    if st.button("3.6 Lọc Wiener", use_container_width=True, type="primary" if st.session_state.current_page == "3.6 Lọc Wiener" else "secondary"):
        st.session_state.current_page = "3.6 Lọc Wiener"
        st.rerun()

# Dùng 1 thẻ div bọc ngoài hr và dùng class margin âm để kéo sát lên trên
st.markdown('<div class="divider-container"><hr class="neumorphic-divider"></div>', unsafe_allow_html=True)

# --- KHU VỰC NỘI DUNG ---
if st.session_state.current_page == "3.4 Hàm Suy Biến":
    # 1. BẢNG ĐIỀU KHIỂN (NẰM NGANG)
    st.markdown("""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px; 
                    box-shadow: 6px 6px 12px rgba(163,177,198,0.6), -6px -6px 12px rgba(255,255,255, 0.9); margin-bottom: 15px;">
            <h3 style="margin-top: 0; margin-bottom: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">Bảng Điều Khiển</h3>
        </div>
    """, unsafe_allow_html=True)
    
    c_mod, c_p1, c_p2, c_noise = st.columns(4)
    with c_mod:
        model_type = st.selectbox("Chọn mô hình suy biến (H):", 
                                  ["Mờ chuyển động (Motion Blur)", "Nhiễu động khí quyển (Turbulence)"])
    with c_p1:
        if model_type == "Mờ chuyển động (Motion Blur)":
            length = st.slider("Chiều dài chuyển động (pixels):", 1, 100, 30)
        else:
            k_val = st.slider("Hệ số nhiễu động k (x0.0001):", 1, 100, 25)
            
    with c_p2:
        if model_type == "Mờ chuyển động (Motion Blur)":
            angle = st.slider("Góc chuyển động (độ):", 0, 180, 45)
            
    with c_noise:
        noise_var = st.slider("Mức độ Nhiễu cộng thêm (η):", 0, 100, 20)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. KHU VỰC HIỂN THỊ ẢNH
    st.markdown("""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px; 
                    box-shadow: inset 4px 4px 8px rgba(163,177,198,0.7), inset -4px -4px 8px rgba(255,255,255, 0.9); margin-bottom: 15px;">
            <h3 style="margin-top: 0; margin-bottom: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">Tiến Trình Suy Biến Ảnh</h3>
        </div>
    """, unsafe_allow_html=True)
        
    # 1. Load ảnh gốc
    img_path = os.path.join(os.path.dirname(__file__), "anh-trang-den-1.webp")
    try:
        pil_img = Image.open(img_path).convert('L')
        orig_img = np.array(pil_img)
        # Resize để xử lý nhanh và vừa màn hình
        if orig_img.shape[0] > 500 or orig_img.shape[1] > 500:
            orig_img = cv2.resize(orig_img, (500, 500))
    except:
        st.error("Không tìm thấy ảnh gốc!")
        orig_img = np.zeros((300, 300), dtype=np.uint8)
        
    img_float = np.float32(orig_img)
    
    # 2. Tính toán hàm suy biến H
    if model_type == "Nhiễu động khí quyển (Turbulence)":
        k = k_val * 0.0001
        F = np.fft.fftshift(np.fft.fft2(img_float))
        rows, cols = img_float.shape
        u = np.arange(-cols/2, cols/2)
        v = np.arange(-rows/2, rows/2)
        U, V = np.meshgrid(u, v)
        H = np.exp(-k * ((U**2 + V**2)**(5/6)))
        G_no_noise = H * F
        degraded_img = np.abs(np.fft.ifft2(np.fft.ifftshift(G_no_noise)))
    else:
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[int(length/2), :] = np.ones(length, dtype=np.float32)
        M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        kernel = kernel / np.sum(kernel)
        degraded_img = cv2.filter2D(img_float, -1, kernel)
        
    # 3. Tạo nhiễu
    noise = np.random.normal(0, noise_var, img_float.shape)
    # Nâng nền nhiễu lên 128 để có thể hiển thị trực quan phần âm/dương của nhiễu
    noise_vis = np.clip(noise + 128, 0, 255).astype(np.uint8) 
    
    # 4. Ảnh cuối cùng (G)
    final_img = degraded_img + noise
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    degraded_img = np.clip(degraded_img, 0, 255).astype(np.uint8)
    
    # --- Bố cục hiển thị ảnh (1 hàng ngang) ---
    c1, arr1, c2, arr2, c3, arr3, c4 = st.columns([3, 1, 3, 1, 3, 1, 3])
    
    with c1:
        st.image(orig_img, caption="1. Ảnh Gốc", use_container_width=True)
        st.latex(r"f(x,y)")
        
    with arr1:
        # Canh giữa dấu mũi tên
        st.markdown("<h3 style='text-align: center; margin-top: 3.5vw; color: #7B61FF;'>➔<br><span style='font-size: 14px;'>H</span></h3>", unsafe_allow_html=True)
        
    with c2:
        st.image(degraded_img, caption="2. Bị Suy Biến", use_container_width=True)
        st.latex(r"H[f(x,y)]")
        
    with arr2:
        st.markdown("<h3 style='text-align: center; margin-top: 4.5vw; color: #7B61FF;'>+</h3>", unsafe_allow_html=True)
        
    with c3:
        st.image(noise_vis, caption="3. Thêm Nhiễu", use_container_width=True)
        st.latex(r"\eta(x,y)")
        
    with arr3:
        st.markdown("<h3 style='text-align: center; margin-top: 4.5vw; color: #7B61FF;'>=</h3>", unsafe_allow_html=True)
        
    with c4:
        st.image(final_img, caption="4. Kết Quả g", use_container_width=True)
        st.latex(r"g(x,y)")
        
    # --- 3. GIẢI PHẪU TOÁN HỌC ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px; 
                    box-shadow: 6px 6px 12px rgba(163,177,198,0.6), -6px -6px 12px rgba(255,255,255, 0.9); margin-bottom: 20px;">
            <h3 style="margin-top: 0; margin-bottom: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">Giải Phẫu Toán Học Của Hàm Suy Biến</h3>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["1. Xung & Tích Chập", "2. Tính Bất Biến (Space-Invariant)", "3. Miền Tần Số (FFT)"])
    
    with tab1:
        # Tạo Xung Dirac (1 điểm sáng)
        impulse = np.zeros((300, 300), dtype=np.uint8)
        impulse[150, 150] = 255
        
        # Áp dụng H lên xung (Tạo đáp ứng xung h)
        if model_type == "Nhiễu động khí quyển (Turbulence)":
            F_imp = np.fft.fftshift(np.fft.fft2(np.float32(impulse)))
            imp_resp = np.abs(np.fft.ifft2(np.fft.ifftshift(H * F_imp)))
            imp_resp = (imp_resp / np.max(imp_resp) * 255).astype(np.uint8)
        else:
            imp_resp = cv2.filter2D(np.float32(impulse), -1, kernel)
            imp_resp = (imp_resp / np.max(imp_resp) * 255).astype(np.uint8)
            
        t1_title1, t1_title2 = st.columns(2)
        with t1_title1:
            st.markdown("**Trường hợp 1:** Hàm $\mathcal{H}$ áp dụng lên **1 Điểm Sáng** (Xung $\delta$)")
        with t1_title2:
            st.markdown("**Trường hợp 2:** Hàm $\mathcal{H}$ áp dụng lên **Ảnh Thực** (Tích phân chập)")

        c_a, c_b, c_c, c_space, c_d, c_e, c_f = st.columns([3, 1, 3, 2, 3, 1, 3])
        with c_a:
            st.image(impulse, caption="Xung Dirac $\delta$ (Một điểm sáng)", use_container_width=True)
        with c_b:
             st.markdown("<h3 style='text-align: center; margin-top: 2vw; color: #7B61FF;'>➔<br><span style='font-size: 14px;'>H</span></h3>", unsafe_allow_html=True)
        with c_c:
            st.image(imp_resp, caption="Đáp ứng Xung $h$ (Vệt/Quầng sáng)", use_container_width=True)
            
        with c_d:
            st.image(orig_img, caption="Ảnh gốc $f$ (Hàng triệu điểm sáng)", use_container_width=True)
        with c_e:
             st.markdown("<h3 style='text-align: center; margin-top: 2vw; color: #7B61FF;'>➔<br><span style='font-size: 14px;'>H</span></h3>", unsafe_allow_html=True)
        with c_f:
            st.image(degraded_img, caption="Ảnh bị mờ $g$ (Hàng triệu quầng sáng $h$)", use_container_width=True)
        
        # Đệm thêm 80px để bù đắp chiều cao bị thiếu so với thanh trượt ở Tab 2
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

    with tab2:
        t2_left, t2_gap, t2_c1, t2_c_gap, t2_c2 = st.columns([7, 2, 3, 1, 3])
        
        with t2_left:
            st.markdown("**So sánh: Hệ Bất biến (Space-Invariant) vs Hệ Biến đổi (Space-Variant)**")
            pos_x = st.slider("Cầm vào đây để Di chuyển vị trí điểm sáng (từ tâm ra rìa):", 0, 299, 150)
        
        imp_move = np.zeros((300, 300), dtype=np.uint8)
        imp_move[150, pos_x] = 255
        
        # Bất biến theo không gian (Invariant)
        if model_type == "Nhiễu động khí quyển (Turbulence)":
            F_m = np.fft.fftshift(np.fft.fft2(np.float32(imp_move)))
            inv_resp = np.abs(np.fft.ifft2(np.fft.ifftshift(H * F_m)))
            inv_resp = (inv_resp / np.max(inv_resp) * 255).astype(np.uint8)
        else:
            inv_resp = cv2.filter2D(np.float32(imp_move), -1, kernel)
            inv_resp = (inv_resp / np.max(inv_resp) * 255).astype(np.uint8)
            
        # Biến đổi theo không gian (Variant - Méo/Mờ hơn khi ra rìa thấu kính)
        dist_from_center = abs(pos_x - 150)
        var_blur_size = int((dist_from_center / 150.0) * 40) + 1 # Nhòe dần từ 1 -> 41
        if var_blur_size % 2 == 0: var_blur_size += 1
        var_resp = cv2.GaussianBlur(imp_move, (var_blur_size, var_blur_size), 0)
        if np.max(var_resp) > 0:
            var_resp = (var_resp / np.max(var_resp) * 255).astype(np.uint8)
            
        with t2_c1:
            st.markdown("**Hệ Biến đổi**")
            st.image(var_resp, use_container_width=True)
        with t2_c2:
            st.markdown("**Hệ Bất biến**")
            st.image(inv_resp, use_container_width=True)

    with tab3:
        st.markdown("**Phép nhân trong Miền Tần Số:** $G(u,v) = H(u,v)F(u,v)$")
        # Tính toán phổ Fourier
        F_mag = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(img_float))))
        F_mag = (F_mag / np.max(F_mag) * 255).astype(np.uint8)
        
        if model_type == "Nhiễu động khí quyển (Turbulence)":
            H_mag_vis = (H * 255).astype(np.uint8)
        else:
            H_fft = np.fft.fft2(kernel, s=img_float.shape)
            H_mag = np.abs(np.fft.fftshift(H_fft))
            H_mag_vis = (H_mag / np.max(H_mag) * 255).astype(np.uint8)
            
        G_mag = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(np.float32(degraded_img)))))
        G_mag = (G_mag / np.max(G_mag) * 255).astype(np.uint8)
        
        t3_sp1, t3_c1, t3_m1, t3_c2, t3_m2, t3_c3, t3_sp2 = st.columns([2.5, 3, 1, 3, 1, 3, 2.5])
        with t3_c1:
            st.image(F_mag, caption="Phổ của Ảnh Gốc $|F(u,v)|$", use_container_width=True)
        with t3_m1:
            st.markdown("<h3 style='text-align: center; margin-top: 3.5vw; color: #7B61FF;'>✖</h3>", unsafe_allow_html=True)
        with t3_c2:
            st.image(H_mag_vis, caption="Phổ của Hàm Suy Biến $|H(u,v)|$", use_container_width=True)
        with t3_m2:
            st.markdown("<h3 style='text-align: center; margin-top: 3.5vw; color: #7B61FF;'>=</h3>", unsafe_allow_html=True)
        with t3_c3:
            st.image(G_mag, caption="Phổ Kết quả $|G(u,v)|$", use_container_width=True)
            
        # Đệm thêm 80px để bù đắp chiều cao bị thiếu so với thanh trượt ở Tab 2
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #111;'>Ước lượng Hàm Suy biến bằng Quan sát (Estimation by Image Observation)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px; color: #555;'>Dựa trên tính <b>Bất biến theo không gian (Space-Invariant)</b>, hàm suy biến của một vùng ảnh con cũng chính là hàm suy biến của toàn bộ ảnh.</p>", unsafe_allow_html=True)
    
    st.info(r"""
    **Công thức cốt lõi:**
    $$
    \hat{H}_s(u,v) = \frac{G_s(u,v)}{\hat{F}_s(u,v)}
    $$
    *Trong đó $G_s$ là phổ của vùng ảnh quan sát bị mờ, và $\hat{F}_s$ là phổ của vùng ảnh gốc tương ứng (được nội suy).*
    """)
    
    st.markdown("**Chọn vùng quan sát trên ảnh:**")
    
    col_x, col_y = st.columns(2)
    with col_x:
        patch_x = st.slider("Tọa độ X vùng quan sát:", 0, img_float.shape[1]-100, 307) # Mặc định trỏ vào mắt
    with col_y:
        patch_y = st.slider("Tọa độ Y vùng quan sát:", 0, img_float.shape[0]-100, 104) # Mặc định trỏ vào mắt
        
    patch_size = 100
    
    # Trích xuất subimage
    g_s = degraded_img[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]
    f_s = img_float[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]
    
    # Tính H_s = G_s / F_s trong miền tần số (sử dụng Hanning window để giảm nhiễu viền)
    window = np.hanning(patch_size)[:, None] * np.hanning(patch_size)[None, :]
    g_s_win = g_s * window
    f_s_win = f_s * window
    
    G_s_freq = np.fft.fft2(g_s_win)
    F_s_freq = np.fft.fft2(f_s_win)
    
    epsilon = 1e-3 # Tránh chia cho 0
    H_s_freq = G_s_freq / (F_s_freq + epsilon)
    
    # Tính phổ biên độ của H_s
    H_s_mag = np.log(1 + np.abs(np.fft.fftshift(H_s_freq)))
    H_s_mag_vis = (H_s_mag / np.max(H_s_mag) * 255).astype(np.uint8)
    
    # Chuẩn hóa lại kiểu dữ liệu để hiển thị không bị trắng xóa (float 0-255 -> uint8)
    g_s_display = np.clip(g_s, 0, 255).astype(np.uint8)
    f_s_display = np.clip(f_s, 0, 255).astype(np.uint8)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Ảnh mờ quan sát $g_s$**")
        st.image(g_s_display, channels="GRAY", use_container_width=True)
    with c2:
        st.markdown("**2. Ảnh gốc nội suy $\hat{f}_s$**")
        st.image(f_s_display, channels="GRAY", use_container_width=True)
    with c3:
        st.markdown("**3. Phổ Hàm suy biến $\hat{H}_s$**")
        st.image(H_s_mag_vis, channels="GRAY", use_container_width=True)

elif st.session_state.current_page == "3.5 Khử Nhiễu Wavelet":
    from wavelet_forward_demo import render_wavelet_demo

    render_wavelet_demo()
    # Phần prototype cũ được giữ lại dưới dạng ghi chú để dễ đối chiếu,
    # nhưng không được thực thi vì renderer mới đã hoàn tất trang 3.5.
    _legacy_wavelet_code = r'''

    import pywt
    
    # 1. BẢNG ĐIỀU KHIỂN
    st.markdown("""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px; 
                    box-shadow: 6px 6px 12px rgba(163,177,198,0.6), -6px -6px 12px rgba(255,255,255, 0.9); margin-bottom: 15px;">
            <h3 style="margin-top: 0; margin-bottom: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">Bảng Điều Khiển Wavelet</h3>
        </div>
    """, unsafe_allow_html=True)
    
    c_w, c_l, c_n, c_t = st.columns(4)
    with c_w:
        wavelet_name = st.selectbox("Chọn bộ lọc Wavelet:", ['haar', 'db2', 'db4', 'sym4', 'coif2'])
    with c_l:
        levels = st.slider("Số cấp phân rã DWT:", 1, 3, 2)
    with c_n:
        noise_level = st.slider("Mức độ nhiễu Gaussian:", 0.0, 100.0, 30.0, step=1.0)
    with c_t:
        threshold = st.slider("Ngưỡng khử nhiễu (Threshold):", 0.0, 150.0, 30.0, step=1.0)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. KHU VỰC HIỂN THỊ ẢNH
    st.markdown("""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px; 
                    box-shadow: inset 4px 4px 8px rgba(163,177,198,0.7), inset -4px -4px 8px rgba(255,255,255, 0.9); margin-bottom: 15px;">
            <h3 style="margin-top: 0; margin-bottom: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">Mô phỏng 3.5: Wavelet Denoising</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Đọc ảnh
    img_path = os.path.join(os.path.dirname(__file__), "anh-trang-den-1.webp")
    if os.path.exists(img_path):
        orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if orig_img.shape[0] > 500:
            orig_img = cv2.resize(orig_img, (500, 500))
            
        img_float = orig_img.astype(np.float32)
        
        # Thêm nhiễu
        np.random.seed(42) # Cố định seed để nhiễu không nhảy lung tung
        noise = np.random.normal(0, noise_level, img_float.shape)
        noisy_img = img_float + noise
        noisy_display = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # --- THỰC HIỆN DWT (Phân rã đa cấp để khử nhiễu) ---
        # coeffs có dạng [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        coeffs = pywt.wavedec2(noisy_img, wavelet_name, level=levels)
        
        # --- KHỬ NHIỄU BẰNG SOFT THRESHOLDING ---
        denoised_coeffs = list(coeffs)
        for i in range(1, len(denoised_coeffs)):
            # Cắt ngưỡng mềm trên các dải chi tiết (LH, HL, HH)
            denoised_coeffs[i] = tuple(pywt.threshold(c, value=threshold, mode='soft') for c in denoised_coeffs[i])
            
        # Tái tạo ảnh (IDWT)
        denoised_img = pywt.waverec2(denoised_coeffs, wavelet_name)
        # Đảm bảo kích thước không bị lệch do padding của DWT
        denoised_img = denoised_img[:img_float.shape[0], :img_float.shape[1]]
        denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
        
        # --- HIỂN THỊ TRỰC QUAN PHÂN RÃ (CHỈ HIỂN THỊ CẤP 1 ĐỂ MINH HỌA) ---
        coeffs_L1 = pywt.dwt2(noisy_img, wavelet_name)
        cA, (cH, cV, cD) = coeffs_L1
        
        def normalize_band(band):
            return np.clip(band, 0, 255).astype(np.uint8)
            
        def normalize_detail(band):
            # Dùng cv2.NORM_MINMAX để sáng rõ đường nét
            norm = cv2.normalize(np.abs(band), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return norm
            
        top_row = np.hstack((normalize_band(cA), normalize_detail(cH)))
        bottom_row = np.hstack((normalize_detail(cV), normalize_detail(cD)))
        dwt_vis = np.vstack((top_row, bottom_row))
        
        # Resize lại cho bằng với ảnh gốc để hiển thị ngang hàng
        dwt_vis = cv2.resize(dwt_vis, (orig_img.shape[1], orig_img.shape[0]))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Ảnh bị nhiễu $x(n)$**")
            st.image(noisy_display, channels="GRAY", use_container_width=True)
        with col2:
            st.markdown("**2. Filterbank DWT (Cấp 1)**")
            st.image(dwt_vis, channels="GRAY", use_container_width=True)
            st.caption("Góc trái trên: LL. 3 góc còn lại: LH, HL, HH")
        with col3:
            st.markdown("**3. Ảnh khử nhiễu IDWT**")
            st.image(denoised_img, channels="GRAY", use_container_width=True)
            
        st.markdown("---")
        st.info(r"""
        **Giải thích sơ đồ Filterbanks (DWT & IDWT):**
        - **Phân rã (Forward DWT):** Tín hiệu $x(n)$ đi qua bộ lọc thông thấp (LL) và thông cao (LH, HL, HH) rồi được lấy mẫu xuống ($\downarrow 2$). Hình ở giữa minh họa rõ nét 4 ma trận kết quả này.
        - **Khử nhiễu (Denoising):** Nhiễu Gaussian thường có tần số cao và biên độ nhỏ, do đó nó nằm lẫn trong 3 dải chi tiết (LH, HL, HH). Bằng thuật toán **Soft Thresholding** (cắt ngưỡng mềm), ta gọt sạch các nhiễu nhỏ hơn Threshold mà vẫn giữ lại được các đường nét biên (edges) lớn.
        - **Tái tạo (Inverse DWT):** Các ma trận sau khi đã "lọc sạch" nhiễu sẽ được lấy mẫu lên ($\uparrow 2$) và đi qua bộ lọc tổng hợp để gộp lại thành bức ảnh cuối cùng mượt mà hơn rất nhiều!
        """)
    '''

elif st.session_state.current_page == "3.6 Lọc Wiener":
    from wiener_filter_demo import render_wiener_demo

    render_wiener_demo()
