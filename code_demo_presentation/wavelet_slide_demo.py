"""Slide-style, image-based demonstration of a three-level 2-D DWT."""

import os
from pathlib import Path

import cv2
import numpy as np
import pywt
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


LEVELS = 3
WAVELET = "db4"
NOISE_SIGMA = 25.0
RNG_SEED = 42


def _font(size, bold=False):
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ) if bold else (
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _preview(array, detail=False, size=(150, 110)):
    """Convert one coefficient band to a readable grayscale PIL image."""
    values = np.asarray(array, dtype=np.float32)
    if detail:
        values = np.abs(values)
    low, high = np.percentile(values, (1, 99)) if values.size else (0, 1)
    if high <= low:
        high = low + 1.0
    image = np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return Image.fromarray(image).resize(size, Image.Resampling.BILINEAR).convert("RGB")


def _detail_tile(details, size=(170, 128)):
    """Show the three 2-D detail bands inside one d_i branch."""
    width, height = size
    tile = Image.new("RGB", size, "#e8edf4")
    labels = ("LH", "HL", "HH")
    band_width = width // 3
    draw = ImageDraw.Draw(tile)
    for index, (label, band) in enumerate(zip(labels, details)):
        band_image = _preview(band, detail=True, size=(band_width - 4, height - 25))
        x = index * band_width + 2
        tile.paste(band_image, (x, 2))
        draw.text((x + 4, height - 20), label, fill="#1f2937", font=_font(13, True))
    return tile


def _gray_tile(array, size=(170, 128)):
    return _preview(array, detail=False, size=size)


def _card(canvas, image, x, y, label, subtitle="", size=(190, 170)):
    width, height = size
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=10, fill="#ffffff", outline="#718096", width=2)
    image = image.resize((width - 16, height - 46), Image.Resampling.BILINEAR)
    canvas.paste(image, (x + 8, y + 8))
    draw.text((x + 8, y + height - 34), label, fill="#111827", font=_font(16, True))
    if subtitle:
        draw.text((x + 8, y + height - 18), subtitle, fill="#4b5563", font=_font(11))


def _box(canvas, x, y, text, width=74, height=38, fill="#f5f7fb"):
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=fill, outline="#4a5568", width=2)
    bbox = draw.textbbox((0, 0), text, font=_font(15, True))
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (width - tw) / 2, y + (height - th) / 2 - 2), text, fill="#111827", font=_font(15, True))


def _arrow(canvas, start, end, color="#4a5568", width=3):
    draw = ImageDraw.Draw(canvas)
    draw.line((*start, *end), fill=color, width=width)
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    head = 10
    points = [
        end,
        (end[0] - head * np.cos(angle - np.pi / 6), end[1] - head * np.sin(angle - np.pi / 6)),
        (end[0] - head * np.cos(angle + np.pi / 6), end[1] - head * np.sin(angle + np.pi / 6)),
    ]
    draw.polygon(points, fill=color)


def _forward_tree(levels_data, input_image):
    """Draw the three-level forward filterbank using real coefficient images."""
    canvas = Image.new("RGB", (1740, 760), "#e5eaf1")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), "FORWARD DWT — three-level iterative filterbank", fill="#111827", font=_font(27, True))
    draw.text((28, 52), "Ảnh 2D: cᵢ = LLᵢ tiếp tục phân rã; dᵢ = {LHᵢ, HLᵢ, HHᵢ}", fill="#374151", font=_font(17))

    # x positions: image input, filter/downsample, output for each level.
    input_x, filter_x, down_x, output_x = 24, 252, 350, 448
    next_filter_x, next_down_x, next_output_x = 676, 774, 872
    last_filter_x, last_down_x, last_output_x = 1100, 1198, 1296
    c_y = {1: 228, 2: 132, 3: 54}
    d_y = {1: 482, 2: 350, 3: 258}
    card_w = 190

    _card(canvas, _gray_tile(input_image), input_x, 336, "x(n)", "ảnh nhiễu vào")

    def branch(start_xy, fy, filter_x_value, down_x_value, output_x_value, output_y, output_image, label, subtitle, detail=False):
        _arrow(canvas, start_xy, (filter_x_value, fy + 19))
        _box(canvas, filter_x_value, fy, "h₀(n)" if label.startswith("c") else "h₁(n)")
        _arrow(canvas, (filter_x_value + 74, fy + 19), (down_x_value, fy + 19))
        _box(canvas, down_x_value, fy, "↓2", width=48)
        _arrow(canvas, (down_x_value + 48, fy + 19), (output_x_value, output_y + 74))
        image = _detail_tile(output_image) if detail else _gray_tile(output_image)
        _card(canvas, image, output_x_value, output_y, label, subtitle)

    input_center = (input_x + card_w, 421)
    branch(input_center, c_y[1], filter_x, down_x, output_x, c_y[1], levels_data[0]["approx"], "c₁", "LL₁")
    branch(input_center, d_y[1], filter_x, down_x, output_x, d_y[1], levels_data[0]["detail"], "d₁", "LH₁ · HL₁ · HH₁", detail=True)

    c1_center = (output_x + card_w, c_y[1] + 85)
    branch(c1_center, c_y[2], next_filter_x, next_down_x, next_output_x, c_y[2], levels_data[1]["approx"], "c₂", "LL₂")
    branch(c1_center, d_y[2], next_filter_x, next_down_x, next_output_x, d_y[2], levels_data[1]["detail"], "d₂", "LH₂ · HL₂ · HH₂", detail=True)

    c2_center = (next_output_x + card_w, c_y[2] + 85)
    branch(c2_center, c_y[3], last_filter_x, last_down_x, last_output_x, c_y[3], levels_data[2]["approx"], "c₃", "LL₃")
    branch(c2_center, d_y[3], last_filter_x, last_down_x, last_output_x, d_y[3], levels_data[2]["detail"], "d₃", "LH₃ · HL₃ · HH₃", detail=True)

    draw.text((1510, 88), "c₃", fill="#111827", font=_font(20, True))
    draw.text((1510, 292), "d₃", fill="#111827", font=_font(20, True))
    draw.text((1510, 358), "d₂", fill="#111827", font=_font(20, True))
    draw.text((1510, 490), "d₁", fill="#111827", font=_font(20, True))
    return canvas


def _inverse_tree(levels_data, original_shape):
    """Draw the three-level inverse filterbank after thresholding."""
    canvas = Image.new("RGB", (1740, 720), "#e5eaf1")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), "INVERSE DWT — three-level synthesis filterbank", fill="#111827", font=_font(27, True))
    draw.text((28, 52), "Các nhánh cᵢ và dᵢ được lấy mẫu lên, lọc tổng hợp rồi cộng lại", fill="#374151", font=_font(17))

    # Source coefficients and reconstructed approximations.
    c3, d3 = levels_data[2]["approx_filtered"], levels_data[2]["detail_filtered"]
    c2, d2 = levels_data[1]["approx_reconstructed"], levels_data[1]["detail_filtered"]
    c1, d1 = levels_data[0]["approx_reconstructed"], levels_data[0]["detail_filtered"]

    x1, x2, x3, x4 = 24, 500, 960, 1420
    top, bottom = 108, 390
    card_w = 190

    _card(canvas, _gray_tile(c3), x1, top, "c₃", "LL₃")
    _card(canvas, _detail_tile(d3), x1, bottom, "d₃", "LH₃ · HL₃ · HH₃")
    _card(canvas, _gray_tile(c2), x2, top + 72, "c₂", "IDWT(c₃,d₃)")
    _card(canvas, _detail_tile(d2), x2, bottom + 36, "d₂", "detail cấp 2")
    _card(canvas, _gray_tile(c1), x3, top + 72, "c₁", "IDWT(c₂,d₂)")
    _card(canvas, _detail_tile(d1), x3, bottom + 36, "d₁", "detail cấp 1")
    _card(canvas, _gray_tile(levels_data[0]["reconstruction"]), x4, top + 72, "x̂(n)", "ảnh sau IDWT")

    def synthesis_pair(source_y, target_y, target_x):
        _arrow(canvas, (x1 + card_w, source_y + 80), (target_x - 180, target_y + 35))
        _box(canvas, target_x - 170, target_y + 16, "↑2", width=48)
        _arrow(canvas, (target_x - 122, target_y + 35), (target_x - 78, target_y + 35))
        _box(canvas, target_x - 72, target_y + 16, "h̃₀/h̃₁", width=70)

    synthesis_pair(top, top + 72, x2)
    synthesis_pair(bottom, bottom + 36, x2)
    _arrow(canvas, (x2 + 190, top + 157), (x3 - 190, top + 157))
    _box(canvas, x3 - 180, top + 138, "↑2", width=48)
    _box(canvas, x3 - 120, top + 138, "⊕", width=42)
    _arrow(canvas, (x3 - 78, top + 157), (x3, top + 157))
    _arrow(canvas, (x2 + 190, bottom + 118), (x3 - 190, bottom + 118))
    _box(canvas, x3 - 180, bottom + 99, "↑2", width=48)
    _box(canvas, x3 - 120, bottom + 99, "h̃₀/h̃₁", width=70)
    _arrow(canvas, (x3 + 190, top + 157), (x4 - 170, top + 157))
    _box(canvas, x4 - 160, top + 138, "↑2", width=48)
    _box(canvas, x4 - 102, top + 138, "⊕", width=42)
    _arrow(canvas, (x4 - 60, top + 157), (x4, top + 157))

    # Green note makes the correspondence with the slide explicit.
    draw.rounded_rectangle((24, 645, 1715, 688), radius=8, fill="#d9f2e3", outline="#4f9d69")
    draw.text((40, 656), "Mỗi phép ⊕ là một bước IDWT: nhánh xấp xỉ cᵢ và nhánh chi tiết dᵢ được ghép lại để tạo cᵢ₋₁.", fill="#155724", font=_font(16, True))
    return canvas


def _prepare_data():
    image_path = os.path.join(os.path.dirname(__file__), "anh-trang-den-1.webp")
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(image_path)
    if max(original.shape) > 512:
        scale = 512 / max(original.shape)
        original = cv2.resize(original, (int(original.shape[1] * scale), int(original.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    original_float = original.astype(np.float32)
    rng = np.random.default_rng(RNG_SEED)
    noisy_float = original_float + rng.normal(0.0, NOISE_SIGMA, original.shape).astype(np.float32)
    coeffs = pywt.wavedec2(noisy_float, WAVELET, level=LEVELS, mode="symmetric")
    sigma_hat = float(np.median(np.abs(coeffs[-1][2])) / 0.6745)
    threshold = sigma_hat * np.sqrt(2 * np.log(noisy_float.size))

    filtered = [coeffs[0]]
    for detail in coeffs[1:]:
        filtered.append(tuple(pywt.threshold(band, threshold, mode="soft") for band in detail))
    denoised = pywt.waverec2(filtered, WAVELET, mode="symmetric")[: original.shape[0], : original.shape[1]]

    # Reconstruct each inverse level so the inverse tree can show real images.
    c2_reconstructed = pywt.idwt2((filtered[0], filtered[1]), WAVELET, mode="symmetric")
    c2_reconstructed = c2_reconstructed[: coeffs[2][0].shape[0], : coeffs[2][0].shape[1]]
    c1_reconstructed = pywt.idwt2((c2_reconstructed, filtered[2]), WAVELET, mode="symmetric")
    c1_reconstructed = c1_reconstructed[: coeffs[3][0].shape[0], : coeffs[3][0].shape[1]]

    levels_data = []
    approx = noisy_float
    for index in range(LEVELS):
        approx, (horizontal, vertical, diagonal) = pywt.dwt2(approx, WAVELET, mode="symmetric")
        levels_data.append({
            "approx": approx,
            "detail": (horizontal, vertical, diagonal),
            "approx_filtered": filtered[0] if index == 2 else None,
            "detail_filtered": filtered[LEVELS - index],
            "approx_reconstructed": c2_reconstructed if index == 1 else c1_reconstructed if index == 0 else filtered[0],
        })

    levels_data[0]["reconstruction"] = denoised
    return original, noisy_float, denoised, threshold, sigma_hat, levels_data


def _psnr(reference, candidate):
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10((255 ** 2) / mse)


def render_wavelet_demo():
    st.markdown("""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px;
                    box-shadow: 6px 6px 12px rgba(163,177,198,0.6), -6px -6px 12px rgba(255,255,255, 0.9); margin-bottom: 15px;">
            <h3 style="margin: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">3.5 Wavelet Denoising — mô phỏng theo sơ đồ filterbank</h3>
        </div>
    """, unsafe_allow_html=True)
    st.info("Cấu hình cố định theo slide: DWT 3 cấp · bộ lọc phân tích h₀/h₁ · ↓2 · bộ lọc tổng hợp h̃₀/h̃₁ · ↑2 · Soft threshold Universal.")

    try:
        original, noisy_float, denoised, threshold, sigma_hat, levels_data = _prepare_data()
    except FileNotFoundError:
        st.error("Không tìm thấy ảnh mẫu anh-trang-den-1.webp")
        return

    noisy_display = np.clip(noisy_float, 0, 255).astype(np.uint8)
    denoised_display = np.clip(denoised, 0, 255).astype(np.uint8)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số cấp DWT", "3")
    c2.metric("Nhiễu Gaussian σ", f"{NOISE_SIGMA:.0f}")
    c3.metric("Ngưỡng Universal λ", f"{threshold:.2f}")
    c4.metric("PSNR sau khử nhiễu", f"{_psnr(original, denoised_display):.2f} dB")

    st.markdown("### 1. Forward DWT — cây phân rã đa cấp")
    st.markdown("Ảnh nhiễu đi qua hai nhánh lọc: **h₀(n)** là thông thấp và **h₁(n)** là thông cao. Sau đó mỗi nhánh được giảm mẫu **↓2**. Chỉ nhánh xấp xỉ **cᵢ** tiếp tục đi sang cấp kế tiếp.")
    forward_image = _forward_tree(levels_data, noisy_display)
    st.image(forward_image, caption="Cây Forward DWT với ảnh thật và các hệ số LL/LH/HL/HH", use_container_width=True)

    st.markdown("### 2. Thresholding — khử nhiễu trên các nhánh chi tiết")
    st.markdown("Các dải **LH, HL, HH** của mỗi `dᵢ` được áp dụng Soft Threshold. Nhánh `LL` được giữ lại để bảo toàn hình dạng và độ sáng tổng thể của ảnh.")
    threshold_cols = st.columns(3)
    with threshold_cols[0]:
        st.image(original, caption="Ảnh sạch f(x,y)", channels="GRAY", use_container_width=True)
    with threshold_cols[1]:
        st.image(noisy_display, caption="Ảnh vào x(n) = f(n) + η(n)", channels="GRAY", use_container_width=True)
    with threshold_cols[2]:
        st.image(denoised_display, caption="Ảnh sau threshold và IDWT", channels="GRAY", use_container_width=True)

    st.markdown("### 3. Inverse DWT — cây tái tạo ảnh")
    st.markdown("Các hệ số `c₃`, `d₃` được lấy mẫu lên **↑2**, lọc tổng hợp rồi cộng để tạo `c₂`; tiếp tục với `d₂`, rồi `d₁` để thu được ảnh đầu ra.")
    inverse_image = _inverse_tree(levels_data, original.shape)
    st.image(inverse_image, caption="Cây Inverse DWT tương ứng với sơ đồ trong slide", use_container_width=True)

    with st.expander("Công thức cốt lõi"):
        st.latex(r"x(n) = f(n) + \\eta(n)")
        st.latex(r"c_i = (x_i * h_0)\\downarrow 2, \\qquad d_i = (x_i * h_1)\\downarrow 2")
        st.latex(r"\\hat c = \\operatorname{sign}(c)\\max(|c|-\\lambda,0)")
        st.latex(r"\\hat f(n) = IDWT\\{c_3,d_3,d_2,d_1\\}")
        st.caption(f"σ̂ được ước lượng từ dải HH₁ bằng MAD: {sigma_hat:.2f}; λ Universal = σ̂√(2 ln N) = {threshold:.2f}.")
