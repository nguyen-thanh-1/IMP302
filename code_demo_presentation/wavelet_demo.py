"""Interactive Wavelet denoising demo for section 3.5.

The rendering is kept separate from app_web.py so the teaching flow is easier
to maintain: add noise -> analyze sub-bands -> threshold details -> reconstruct.
"""

import os

import cv2
import numpy as np
import pywt
import streamlit as st


def _band_to_image(band, detail=False):
    """Convert a wavelet band to an easy-to-read grayscale preview."""
    values = np.asarray(band, dtype=np.float32)
    if detail:
        values = np.abs(values)
    low, high = np.percentile(values, (1, 99)) if values.size else (0, 1)
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _psnr(reference, candidate):
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)


def _section_heading(text, inset=False):
    shadow = (
        "inset 4px 4px 8px rgba(163,177,198,0.7), "
        "inset -4px -4px 8px rgba(255,255,255, 0.9)"
        if inset
        else "6px 6px 12px rgba(163,177,198,0.6), -6px -6px 12px rgba(255,255,255, 0.9)"
    )
    st.markdown(
        f"""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px;
                    box-shadow: {shadow}; margin-bottom: 15px;">
            <h3 style="margin: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">{text}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_wavelet_demo():
    """Render the complete Streamlit UI for section 3.5."""
    _section_heading("3.5 Khử nhiễu ảnh bằng Wavelet")
    st.markdown(
        """
        <div style="background: rgba(255,255,255,.38); padding: 10px 16px; border-radius: 12px; margin-bottom: 12px;">
        <b>Ý tưởng:</b> DWT tách ảnh thành một dải xấp xỉ <b>LL</b> và các dải chi tiết
        <b>LH, HL, HH</b>. Nhiễu Gaussian thường tạo ra các hệ số chi tiết nhỏ, vì vậy ta
        ngưỡng hóa các hệ số này rồi dùng IDWT để tái tạo ảnh.
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_1, control_2, control_3, control_4 = st.columns(4)
    with control_1:
        wavelet_name = st.selectbox(
            "Bộ lọc / wavelet", ["haar", "db2", "db4", "sym4", "coif2"], index=2
        )
    with control_2:
        levels = st.slider("Số cấp phân rã", 1, 3, 2)
    with control_3:
        noise_level = st.slider("Nhiễu Gaussian σ", 0.0, 80.0, 25.0, step=1.0)
    with control_4:
        threshold_mode = st.selectbox("Cách chọn ngưỡng", ["Tự động - Universal", "Thủ công"])

    control_5, control_6, control_7, control_8 = st.columns(4)
    with control_5:
        threshold_type = st.selectbox("Kiểu threshold", ["Soft (mềm)", "Hard (cứng)"])
    with control_6:
        if threshold_mode == "Thủ công":
            manual_threshold = st.slider("Ngưỡng λ thủ công", 0.0, 120.0, 25.0, step=1.0)
        else:
            manual_threshold = 0.0
    with control_7:
        show_level = st.slider("Cấp DWT muốn xem", 1, levels, levels)
    with control_8:
        st.markdown(
            "<div style='padding-top: 28px; color: #555;'>Seed nhiễu: <b>42</b> (ổn định)</div>",
            unsafe_allow_html=True,
        )

    image_path = os.path.join(os.path.dirname(__file__), "anh-trang-den-1.webp")
    if not os.path.exists(image_path):
        st.error("Không tìm thấy ảnh mẫu anh-trang-den-1.webp")
        return

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        st.error("Không thể đọc ảnh mẫu.")
        return

    max_side = 512
    if max(original.shape) > max_side:
        scale = max_side / max(original.shape)
        original = cv2.resize(
            original,
            (int(original.shape[1] * scale), int(original.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    original_float = original.astype(np.float32)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_level, original_float.shape).astype(np.float32)
    noisy_float = original_float + noise
    noisy_display = np.clip(noisy_float, 0, 255).astype(np.uint8)

    # PyWavelets returns [cA_n, (cH_n,cV_n,cD_n), ..., (cH_1,cV_1,cD_1)].
    coeffs = pywt.wavedec2(noisy_float, wavelet_name, level=levels, mode="symmetric")
    finest_diagonal = coeffs[-1][2]
    sigma_hat = (
        float(np.median(np.abs(finest_diagonal)) / 0.6745)
        if finest_diagonal.size
        else 0.0
    )
    universal_threshold = sigma_hat * np.sqrt(2.0 * np.log(noisy_float.size))
    threshold = universal_threshold if threshold_mode.startswith("Tự động") else manual_threshold
    threshold_kind = "soft" if threshold_type.startswith("Soft") else "hard"

    filtered_coeffs = [coeffs[0]]
    for detail_bands in coeffs[1:]:
        filtered_coeffs.append(
            tuple(pywt.threshold(band, threshold, mode=threshold_kind) for band in detail_bands)
        )

    denoised_float = pywt.waverec2(filtered_coeffs, wavelet_name, mode="symmetric")
    denoised_float = denoised_float[: original_float.shape[0], : original_float.shape[1]]
    denoised = np.clip(denoised_float, 0, 255).astype(np.uint8)

    # Build a one-level-at-a-time pyramid for the teaching view.
    pyramid = []
    approximation = noisy_float
    for _level in range(1, levels + 1):
        approximation, (horizontal, vertical, diagonal) = pywt.dwt2(
            approximation, wavelet_name, mode="symmetric"
        )
        pyramid.append((approximation, horizontal, vertical, diagonal))
    selected_bands = pyramid[show_level - 1]

    _section_heading("Luồng xử lý: thêm nhiễu → DWT → threshold → IDWT", inset=True)
    flow_1, flow_2, flow_3, flow_4 = st.columns(4)
    with flow_1:
        st.image(original, caption="1. Ảnh sạch f(x, y)", channels="GRAY", use_container_width=True)
    with flow_2:
        st.image(noisy_display, caption="2. Ảnh nhiễu x(n)", channels="GRAY", use_container_width=True)
    with flow_3:
        st.image(denoised, caption="3. Ảnh sau ngưỡng hóa", channels="GRAY", use_container_width=True)
    with flow_4:
        removed = np.clip(noisy_float - denoised_float + 128.0, 0, 255).astype(np.uint8)
        st.image(removed, caption="4. Thành phần bị loại", channels="GRAY", use_container_width=True)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("σ ước lượng từ HH₁", f"{sigma_hat:.2f}")
    metric_2.metric("Ngưỡng λ đang dùng", f"{threshold:.2f}")
    metric_3.metric("PSNR ảnh nhiễu", f"{_psnr(original, noisy_display):.2f} dB")
    metric_4.metric("PSNR sau khử nhiễu", f"{_psnr(original, denoised):.2f} dB")

    st.markdown("---")
    st.markdown(f"### Phân rã DWT cấp {show_level}: bốn dải tần sau lọc phân tích")
    band_columns = st.columns(4)
    captions = [
        "LL — xấp xỉ / tần số thấp",
        "LH — chi tiết ngang",
        "HL — chi tiết dọc",
        "HH — chi tiết chéo",
    ]
    for column, band, caption, is_detail in zip(
        band_columns, selected_bands, captions, (False, True, True, True)
    ):
        with column:
            st.image(
                _band_to_image(band, is_detail),
                caption=caption,
                channels="GRAY",
                use_container_width=True,
            )
    st.caption("Các dải LH, HL, HH được hiển thị theo |hệ số| và tự co giãn tương phản để nhìn rõ cấu trúc biên.")

    with st.expander("Xem năng lượng của các dải chi tiết"):
        energy_rows = []
        for index, detail_bands in enumerate(reversed(coeffs[1:]), start=1):
            for name, band in zip(("LH", "HL", "HH"), detail_bands):
                energy_rows.append(
                    {
                        "Cấp (1 là mịn nhất)": index,
                        "Dải": name,
                        "Năng lượng trước lọc": round(float(np.mean(band**2)), 2),
                        "Tỷ lệ hệ số bị triệt": f"{100.0 * np.mean(np.abs(band) <= threshold):.1f}%",
                    }
                )
        st.dataframe(energy_rows, use_container_width=True, hide_index=True)

    with st.expander("Công thức và cách thuyết trình"):
        st.latex(r"x(n) = f(n) + \eta(n)")
        st.latex(r"[LL_j, LH_j, HL_j, HH_j] = DWT_j\{x(n)\}")
        st.latex(r"\hat{c} = \operatorname{soft}(c,\lambda) = \operatorname{sign}(c)\max(|c|-\lambda,0)")
        st.latex(r"\hat{f}(n) = IDWT\{LL,\hat{LH},\hat{HL},\hat{HH}\}")
        st.markdown(
            "- **DWT thuận:** mỗi cấp lọc thông thấp/thông cao rồi lấy mẫu xuống 2, tạo LL và ba dải chi tiết.\n"
            "- **Threshold:** chỉ xử lý các dải chi tiết; LL được giữ lại để bảo toàn hình dạng tổng thể.\n"
            "- **IDWT ngược:** lấy mẫu lên và cộng các nhánh để tái tạo ảnh. Soft threshold thường cho ảnh mượt hơn; Hard threshold giữ biên mạnh hơn nhưng dễ tạo gợn."
        )
