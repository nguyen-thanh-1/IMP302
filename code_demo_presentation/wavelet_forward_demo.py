"""Step-by-step forward DWT demonstration for the Streamlit presentation."""

import os
import base64
import html
from io import BytesIO
import uuid

import cv2
import numpy as np
import pywt
import streamlit as st
from PIL import Image


LEVELS = 3
WAVELET = "db4"


def _band_preview(band, detail=False, size=(220, 150)):
    values = np.asarray(band, dtype=np.float32)
    if detail:
        values = np.abs(values)
    low, high = np.percentile(values, (1, 99)) if values.size else (0, 1)
    if high <= low:
        high = low + 1.0
    image = np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _detail_mosaic(details, size=(220, 150)):
    width, height = size
    tile_width = width // 3
    mosaic = np.zeros((height, width), dtype=np.uint8)
    for index, band in enumerate(details):
        tile = _band_preview(band, detail=True, size=(tile_width - 4, height))
        x0 = index * tile_width + 2
        mosaic[:, x0:x0 + tile.shape[1]] = tile
    return mosaic


def _image_data_uri(image):
    """Encode an already normalized grayscale image for the fixed HTML layout."""
    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _html_image_card(image, caption, detail=False, size=(210, 142), card_class=""):
    preview = _detail_mosaic(image, size=size) if detail else _band_preview(image, size=size)
    # Dùng một ảnh độ phân giải lớn riêng cho modal; không phóng đại thumbnail.
    modal_size = (1200, 780) if detail else (1000, 680)
    modal_preview = _detail_mosaic(image, size=modal_size) if detail else _band_preview(image, size=modal_size)
    image_uri = _image_data_uri(preview)
    modal_uri = _image_data_uri(modal_preview)
    zoom_id = "wavelet_zoom_" + uuid.uuid4().hex
    safe_caption = html.escape(caption, quote=True)
    return (
        f"<div class='forward-image-card {card_class}'>"
        f"<input class='zoom-toggle' type='checkbox' id='{zoom_id}'>"
        f"<label class='zoom-trigger' for='{zoom_id}' title='Bấm để phóng to'>"
        f"<img src='{image_uri}' alt='{safe_caption}'>"
        f"<div class='forward-image-caption'>{safe_caption}</div>"
        "</label>"
        f"<div class='zoom-modal'>"
        f"<label class='zoom-backdrop' for='{zoom_id}' title='Bấm để đóng'>"
        f"<div class='zoom-dialog'><img src='{modal_uri}' alt='{safe_caption}'>"
        f"<span class='zoom-close'>×</span><div class='zoom-modal-caption'>{safe_caption}</div></div>"
        "</label></div>"
        "</div>"
    )


def _load_clean_image():
    image_path = os.path.join(os.path.dirname(__file__), "anh-trang-den-1.webp")
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(image_path)
    if max(original.shape) > 512:
        scale = 512 / max(original.shape)
        original = cv2.resize(
            original,
            (int(original.shape[1] * scale), int(original.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return original


def _add_noise(image, noise_type, noise_level, seed):
    """Create a deterministic noisy input so the whole demo is reproducible."""
    clean = image.astype(np.float32)
    rng = np.random.default_rng(int(seed))

    if noise_type == "Gaussian":
        noisy = clean + rng.normal(0.0, float(noise_level), clean.shape)
    elif noise_type == "Muối tiêu":
        noisy = clean.copy()
        density = float(noise_level)
        mask = rng.random(clean.shape)
        noisy[mask < density / 2.0] = 0.0
        noisy[(mask >= density / 2.0) & (mask < density)] = 255.0
    else:  # Speckle
        strength = float(noise_level)
        noisy = clean + clean * rng.normal(0.0, strength, clean.shape)

    return np.clip(noisy, 0, 255).astype(np.uint8)


def _load_forward_data(noise_config=None):
    clean = _load_clean_image()
    if noise_config is None:
        noisy = clean.copy()
    else:
        noisy = _add_noise(
            clean,
            noise_config["noise_type"],
            noise_config["noise_level"],
            noise_config["seed"],
        )

    stages = []
    current = noisy.astype(np.float32)
    for level in range(1, LEVELS + 1):
        approximation, (horizontal, vertical, diagonal) = pywt.dwt2(
            current, WAVELET, mode="symmetric"
        )
        stages.append(
            {
                "level": level,
                "input": current,
                "approximation": approximation,
                "details": (horizontal, vertical, diagonal),
            }
        )
        current = approximation
    return clean, noisy, stages


def _estimate_noise_sigma(forward_stages):
    finest_hh = np.asarray(forward_stages[0]["details"][2], dtype=np.float32)
    return max(float(np.median(np.abs(finest_hh)) / 0.6745), 1e-6)


def _denoise_details(forward_stages, method, threshold_factor):
    """Threshold LH/HL/HH at every level and return details plus thresholds."""
    sigma = _estimate_noise_sigma(forward_stages)
    denoised_details = []
    threshold_info = []

    for stage in forward_stages:
        filtered_bands = []
        band_thresholds = []
        for band in stage["details"]:
            values = np.asarray(band, dtype=np.float32)
            if method == "Adaptive (BayesShrink)":
                signal_variance = max(float(np.var(values)) - sigma**2, 1e-6)
                threshold = float(threshold_factor) * sigma**2 / np.sqrt(signal_variance)
            else:
                threshold = float(threshold_factor) * sigma * np.sqrt(2.0 * np.log(max(values.size, 2)))

            mode = "soft" if method == "Soft threshold" else "hard"
            filtered_bands.append(pywt.threshold(values, threshold, mode=mode))
            band_thresholds.append(threshold)

        denoised_details.append(tuple(filtered_bands))
        threshold_info.append(band_thresholds)

    return denoised_details, sigma, threshold_info


def _filter_box(title, subtitle):
    st.markdown(
        """
        <div class="wavelet-filter-box">
            <div class="wavelet-filter-title">{title}</div>
            <div class="wavelet-filter-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _image_card(image, caption, detail=False, display_width=180):
    st.image(
        _detail_mosaic(image, size=(180, 120)) if detail else _band_preview(image, size=(180, 120)),
        caption=caption,
        channels="GRAY",
        width=display_width,
    )


def _render_process_area(stage):
    level = stage["level"]
    st.markdown(
        f"<div class='area-title'>Ô A — Đang mô phỏng Level {level}</div>",
        unsafe_allow_html=True,
    )
    input_caption = "Input ảnh thật"
    output_1 = f"Output 1: c{level}(n) = LL{level}"
    output_2 = f"Output 2: d{level}(n) = LH{level}, HL{level}, HH{level}"
    process_html = f"""
    <div class="forward-stage-scroll">
      <div class="forward-stage">
        <div class="forward-input">{_html_image_card(stage["input"], input_caption, size=(210, 142))}</div>
        <div class="forward-branches">
          <div class="forward-branch">
            <div class="forward-filter"><b>h₀(n)</b><small>Low-pass · thông thấp</small></div>
            <div class="forward-arrow">→</div>
            <div class="forward-down"><b>↓2</b><small>downsampling 2</small></div>
            <div class="forward-arrow output-arrow">→ output 1</div>
          </div>
          <div class="forward-branch">
            <div class="forward-filter"><b>h₁(n)</b><small>High-pass · thông cao</small></div>
            <div class="forward-arrow">→</div>
            <div class="forward-down"><b>↓2</b><small>downsampling 2</small></div>
            <div class="forward-arrow output-arrow">→ output 2</div>
          </div>
        </div>
        <div class="forward-outputs">
          {_html_image_card(stage["approximation"], output_1, size=(210, 142))}
          {_html_image_card(stage["details"], output_2, detail=True, size=(210, 142))}
        </div>
      </div>
    </div>
    """
    st.markdown(process_html, unsafe_allow_html=True)


def _render_storage_area(stored):
    st.markdown("<div class='area-title'>Ô B — Kết quả đã lưu</div>", unsafe_allow_html=True)
    if not stored:
        return

    cards = []
    for item in stored:
        cards.append(
            _html_image_card(
                item["data"],
                item["caption"],
                detail=item["kind"] == "detail",
                size=(210, 142),
            )
        )
    st.markdown(
        f"<div class='stored-scroll'><div class='stored-list'>{''.join(cards)}</div></div>",
        unsafe_allow_html=True,
    )


def _crop_to_shape(array, shape):
    return array[: shape[0], : shape[1]]


def _build_inverse_stages(original, forward_stages, denoised_details=None):
    """Create the three real IDWT steps from c3,d3,d2,d1."""
    c3 = forward_stages[2]["approximation"]
    details = denoised_details if denoised_details is not None else [stage["details"] for stage in forward_stages]
    d3 = details[2]
    d2 = details[1]
    d1 = details[0]

    c2 = pywt.idwt2((c3, d3), WAVELET, mode="symmetric")
    c2 = _crop_to_shape(c2, forward_stages[1]["approximation"].shape)
    c1 = pywt.idwt2((c2, d2), WAVELET, mode="symmetric")
    c1 = _crop_to_shape(c1, forward_stages[0]["approximation"].shape)
    reconstructed = pywt.idwt2((c1, d1), WAVELET, mode="symmetric")
    reconstructed = _crop_to_shape(reconstructed, original.shape)

    return [
        {
            "level": 3,
            "approximation": c3,
            "details": d3,
            "output": c2,
            "output_label": "c₂(n)",
            "output_caption": "IDWT(c₃, d₃) → c₂(n)",
        },
        {
            "level": 2,
            "approximation": c2,
            "details": d2,
            "output": c1,
            "output_label": "c₁(n)",
            "output_caption": "IDWT(c₂, d₂) → c₁(n)",
        },
        {
            "level": 1,
            "approximation": c1,
            "details": d1,
            "output": reconstructed,
            "output_label": "x̂(n)",
            "output_caption": "IDWT(c₁, d₁) → ảnh tái tạo",
        },
    ]


def _render_inverse_process_area(stage):
    level = stage["level"]
    next_level = level - 1
    st.markdown(
        f"<div class='area-title'>Ô A — Đang tái tạo Inverse Level {level}</div>",
        unsafe_allow_html=True,
    )
    html = f"""
    <div class="forward-stage-scroll">
      <div class="inverse-stage">
        <div class="inverse-inputs">
          {_html_image_card(stage["approximation"], f"c{level}(n) · LL{level}", size=(210, 142))}
          {_html_image_card(stage["details"], f"d{level}(n) · LH{level}, HL{level}, HH{level}", detail=True, size=(210, 142))}
        </div>
        <div class="inverse-branches">
          <div class="forward-branch">
            <div class="inverse-source-label">c{level}(n)</div>
            <div class="forward-arrow">→</div>
            <div class="forward-down"><b>↑2</b><small>upsampling 2</small></div>
            <div class="forward-arrow output-arrow">→ h̃₀(n) → ⊕</div>
          </div>
          <div class="forward-branch">
            <div class="inverse-source-label">d{level}(n)</div>
            <div class="forward-arrow">→</div>
            <div class="forward-down"><b>↑2</b><small>upsampling 2</small></div>
            <div class="forward-arrow output-arrow">→ h̃₁(n) → ⊕</div>
          </div>
        </div>
        <div class="inverse-output">
          <div class="inverse-sum">⊕<small>c{next_level}(n)</small></div>
          {_html_image_card(stage["output"], f"{stage['output_label']} · {stage['output_caption']}", size=(210, 142))}
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_inverse_storage_area(forward_stages, inverse_step, denoised_details=None):
    st.markdown("<div class='area-title'>Ô B — Các hệ số lấy từ Forward DWT</div>", unsafe_allow_html=True)
    details = denoised_details if denoised_details is not None else [stage["details"] for stage in forward_stages]
    source_items = [
        ("c₃(n) · LL₃", forward_stages[2]["approximation"], False, inverse_step == 0),
        ("d₃(n) · LH₃, HL₃, HH₃", details[2], True, inverse_step == 0),
        ("d₂(n) · LH₂, HL₂, HH₂", details[1], True, inverse_step == 1),
        ("d₁(n) · LH₁, HL₁, HH₁", details[0], True, inverse_step == 2),
    ]
    cards = []
    for caption, data, detail, active in source_items:
        active_class = " source-active" if active else ""
        cards.append(
            f"<div class='source-card{active_class}'>"
            f"{_html_image_card(data, caption, detail=detail, size=(210, 142))}"
            f"</div>"
        )
    st.markdown(
        f"<div class='stored-scroll'><div class='stored-list'>{''.join(cards)}</div></div>",
        unsafe_allow_html=True,
    )


def _render_inverse_demo(input_image, forward_stages, denoised_details):
    inverse_stages = _build_inverse_stages(input_image, forward_stages, denoised_details)
    if "inverse_step" not in st.session_state:
        st.session_state.inverse_step = 0
    step = st.session_state.inverse_step

    control_left, control_mid, control_right = st.columns([1.2, 2.6, 1.2])
    with control_left:
        if st.button("← Về Forward", use_container_width=True):
            st.session_state.dwt_mode = "forward"
            st.session_state.forward_step = LEVELS
            st.rerun()
    with control_mid:
        st.progress(step / LEVELS)
        st.markdown(f"<div style='text-align:center; color:#4b5563;'>Inverse bước {step}/{LEVELS}</div>", unsafe_allow_html=True)
    with control_right:
        if st.button(
            "Next →" if step < LEVELS else "Đã hoàn tất",
            use_container_width=True,
            type="primary",
            disabled=step >= LEVELS,
        ):
            st.session_state.inverse_step += 1
            st.rerun()

    if step < LEVELS:
        _render_inverse_process_area(inverse_stages[step])
    else:
        st.markdown("<div class='area-title'>Ô A — Inverse DWT đã hoàn tất</div>", unsafe_allow_html=True)
        final_cards = (
            _html_image_card(
                input_image,
                "Ảnh nhiễu đầu vào · x(n)",
                size=(360, 240),
                card_class="final-result-card",
            )
            + _html_image_card(
                inverse_stages[-1]["output"],
                "Ảnh tái tạo Inverse · x̂(n)",
                size=(360, 240),
                card_class="final-result-card",
            )
        )
        st.markdown(
            f"<div class='final-result-scroll'><div class='final-result-list'>{final_cards}</div></div>",
            unsafe_allow_html=True,
        )

    _render_inverse_storage_area(forward_stages, min(step, LEVELS - 1), denoised_details)


def _render_setup(clean_image):
    st.markdown("<div class='area-title'>Thiết lập ảnh nhiễu và phương pháp khử nhiễu</div>", unsafe_allow_html=True)
    st.markdown("Chọn tham số một lần trước khi bắt đầu. Sau khi bấm bắt đầu, các tham số sẽ được khóa.")

    left, right = st.columns(2)
    with left:
        noise_type = st.selectbox("Loại nhiễu đầu vào", ["Gaussian", "Muối tiêu", "Speckle"])
        if noise_type == "Gaussian":
            noise_level = st.slider("Độ lệch chuẩn σ", 0.0, 60.0, 20.0, step=1.0)
        elif noise_type == "Muối tiêu":
            noise_level = st.slider("Mật độ nhiễu", 0.0, 0.30, 0.05, step=0.01)
        else:
            noise_level = st.slider("Cường độ speckle", 0.0, 1.0, 0.15, step=0.01)
        seed = st.number_input("Seed nhiễu", min_value=0, max_value=9999, value=42, step=1)
    with right:
        method = st.selectbox(
            "Phương pháp khử nhiễu wavelet",
            ["Soft threshold", "Hard threshold", "Adaptive (BayesShrink)"],
        )
        threshold_factor = st.slider(
            "Hệ số ngưỡng",
            0.2,
            3.0,
            1.0 if method != "Adaptive (BayesShrink)" else 1.2,
            step=0.1,
        )

    preview_config = {
        "noise_type": noise_type,
        "noise_level": noise_level,
        "seed": seed,
        "method": method,
        "threshold_factor": threshold_factor,
    }
    preview = _add_noise(clean_image, noise_type, noise_level, seed)
    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.image(clean_image, caption="Ảnh sạch tham chiếu", channels="GRAY", width=360)
    with preview_right:
        st.image(preview, caption="Ảnh nhiễu sẽ đưa vào Forward", channels="GRAY", width=360)

    if st.button("Bắt đầu Forward DWT →", type="primary", use_container_width=True):
        st.session_state.wavelet_config = preview_config
        st.session_state.dwt_mode = "forward"
        st.session_state.forward_step = 0
        st.session_state.inverse_step = 0
        st.rerun()


def _render_denoise_demo(noisy_image, forward_stages, config):
    denoised_details, sigma, threshold_info = _denoise_details(
        forward_stages,
        config["method"],
        config["threshold_factor"],
    )
    st.markdown("<div class='area-title'>Wavelet Denoising — lọc các hệ số chi tiết</div>", unsafe_allow_html=True)
    st.markdown(
        f"**{config['method']}** · σ ước lượng = **{sigma:.2f}** · hệ số ngưỡng = **{config['threshold_factor']:.1f}**"
    )

    cards = []
    for index, stage in enumerate(forward_stages):
        before = _html_image_card(
            stage["details"],
            f"d{index + 1}(n) trước lọc",
            detail=True,
            size=(260, 175),
        )
        after = _html_image_card(
            denoised_details[index],
            f"d{index + 1}(n) sau lọc",
            detail=True,
            size=(260, 175),
        )
        thresholds = " / ".join(f"{value:.1f}" for value in threshold_info[index])
        cards.append(
            f"<div class='denoise-level-card'><div class='denoise-level-title'>Level {index + 1}</div>"
            f"<div class='denoise-pair'>{before}{after}</div>"
            f"<div class='denoise-threshold'>Ngưỡng LH / HL / HH: {thresholds}</div></div>"
        )

    st.markdown(
        f"<div class='denoise-scroll'><div class='denoise-level-list'>{''.join(cards)}</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("Tiếp tục Inverse DWT →", type="primary", use_container_width=True):
        st.session_state.dwt_mode = "inverse"
        st.session_state.inverse_step = 0
        st.rerun()

    return denoised_details


def render_wavelet_demo():
    if "dwt_mode" not in st.session_state or "wavelet_config" not in st.session_state:
        st.session_state.dwt_mode = "setup"
    mode = st.session_state.dwt_mode

    st.markdown(
        """
        <style>
        .area-title {
            background: #e0e5ec;
            border-radius: 14px;
            padding: 13px 16px;
            margin: 8px 0 10px;
            text-align: center;
            color: #111827;
            font-size: 21px;
            font-weight: 800;
            box-shadow: 5px 5px 10px rgba(163,177,198,.55), -5px -5px 10px rgba(255,255,255,.85);
        }
        .wavelet-filter-box {
            background: #f8fafc;
            border: 2px solid #8795aa;
            border-radius: 10px;
            padding: 7px 5px;
            margin: 5px 0;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(163,177,198,.45);
        }
        .wavelet-filter-title { font-size: 17px; font-weight: 800; color: #111827; }
        .wavelet-filter-subtitle { font-size: 10px; color: #4b5563; margin-top: 3px; }
        .filter-stack-label { text-align: center; font-weight: 700; color: #4b5563; margin-bottom: 5px; }
        .flow-arrow { text-align: center; font-size: 16px; font-weight: 800; color: #7058e8; margin: 0; white-space: nowrap; }
        .forward-stage-scroll, .stored-scroll {
            width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 12px 0 18px;
            scrollbar-color: #8b7cf6 #d8dde7;
        }
        .forward-stage {
            width: 1120px;
            min-width: 1120px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 230px 570px 230px;
            column-gap: 42px;
            align-items: center;
        }
        .forward-input { align-self: center; }
        .forward-branches, .forward-outputs { display: grid; gap: 26px; }
        .forward-branch {
            display: grid;
            grid-template-columns: 190px 34px 125px 150px;
            align-items: center;
            min-height: 142px;
        }
        .forward-filter, .forward-down {
            background: #f8fafc;
            border: 2px solid #8795aa;
            border-radius: 10px;
            padding: 12px 7px;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(163,177,198,.45);
            min-height: 58px;
        }
        .forward-filter b, .forward-down b { display: block; font-size: 19px; color: #111827; }
        .forward-filter small, .forward-down small { display: block; font-size: 11px; color: #4b5563; margin-top: 4px; }
        .forward-arrow { text-align: center; font-size: 22px; font-weight: 800; color: #7058e8; white-space: nowrap; }
        .output-arrow { font-size: 16px; }
        .forward-image-card { width: 210px; text-align: center; color: #6b7280; }
        .zoom-toggle { display: none; }
        .zoom-trigger { display: block; cursor: zoom-in; }
        .forward-image-card img { display: block; width: 210px; height: 142px; object-fit: cover; border-radius: 8px; margin: 0 auto 6px; box-shadow: 1px 2px 5px rgba(31,41,55,.18); }
        .forward-image-caption { font-size: 13px; line-height: 1.35; min-height: 34px; }
        .zoom-modal { display: none; position: fixed; inset: 0; z-index: 999999; }
        .zoom-toggle:checked ~ .zoom-modal { display: flex; align-items: center; justify-content: center; }
        .zoom-backdrop { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(15,23,42,.82); cursor: zoom-out; padding: 30px; }
        .zoom-dialog { position: relative; max-width: 94vw; max-height: 94vh; padding: 18px 18px 12px; border-radius: 14px; background: #f8fafc; box-shadow: 0 18px 60px rgba(0,0,0,.45); cursor: default; }
        .zoom-dialog img { display: block; width: min(82vw, 1200px); height: auto; max-height: 82vh; object-fit: contain; border-radius: 8px; }
        .zoom-close { position: absolute; right: 8px; top: 2px; color: #111827; font-size: 28px; line-height: 28px; font-weight: 800; cursor: zoom-out; }
        .zoom-modal-caption { text-align: center; color: #374151; font-size: 14px; margin-top: 8px; }
        .stored-list { width: max-content; min-width: 100%; display: flex; justify-content: center; gap: 30px; }
        .stored-list .forward-image-card { flex: 0 0 210px; }
        .final-result-scroll { width: 100%; overflow-x: auto; padding: 18px 0 24px; }
        .final-result-list { width: max-content; min-width: 100%; display: flex; justify-content: center; gap: 46px; }
        .final-result-list .forward-image-card { width: 360px; flex: 0 0 360px; }
        .final-result-list .forward-image-card img { width: 360px; height: 240px; }
        .final-result-list .forward-image-caption { font-size: 15px; min-height: 0; }
        .inverse-stage {
            width: 1120px;
            min-width: 1120px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 230px 570px 230px;
            column-gap: 42px;
            align-items: center;
        }
        .inverse-inputs, .inverse-output { display: grid; gap: 26px; align-items: center; }
        .inverse-source-label {
            background: #f8fafc;
            border: 2px solid #8795aa;
            border-radius: 10px;
            min-height: 58px;
            padding: 15px 7px;
            text-align: center;
            font-weight: 800;
            font-size: 18px;
            color: #111827;
        }
        .inverse-sum { text-align: center; font-size: 32px; font-weight: 900; color: #7058e8; margin-bottom: -10px; }
        .inverse-sum small { display: block; font-size: 14px; color: #4b5563; font-weight: 700; }
        .source-card { flex: 0 0 210px; border-radius: 10px; padding: 6px; opacity: .72; }
        .source-active { opacity: 1; outline: 3px solid #7058e8; background: #eeeaff; }
        .source-status { text-align: center; color: #6b7280; font-size: 12px; margin-top: 2px; }
        .denoise-scroll { width: 100%; overflow-x: auto; padding: 12px 0 20px; }
        .denoise-level-list { width: max-content; min-width: 100%; display: flex; justify-content: center; gap: 24px; }
        .denoise-level-card { width: 548px; flex: 0 0 548px; padding: 12px; border-radius: 12px; background: #e0e5ec; box-shadow: 4px 4px 10px rgba(163,177,198,.45); }
        .denoise-level-title { text-align: center; font-weight: 800; color: #111827; margin-bottom: 8px; }
        .denoise-pair { display: flex; gap: 18px; justify-content: center; }
        .denoise-pair .forward-image-card { width: 260px; flex: 0 0 260px; }
        .denoise-pair .forward-image-card img { width: 260px; height: 175px; }
        .denoise-pair .forward-image-caption { min-height: 0; font-size: 13px; }
        .denoise-threshold { text-align: center; color: #4b5563; font-size: 12px; margin-top: 8px; }
        @media (max-width: 1180px) {
            .forward-stage { margin-left: 18px; margin-right: 18px; }
            .stored-list { justify-content: flex-start; padding: 0 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    page_mode = "Thiết lập" if mode == "setup" else ("Forward" if mode == "forward" else ("Wavelet Denoising" if mode == "denoise" else "Inverse"))
    st.markdown(
        f"""
        <div style="background-color: #e0e5ec; padding: 15px; border-radius: 15px;
                    box-shadow: 6px 6px 12px rgba(163,177,198,0.6), -6px -6px 12px rgba(255,255,255, 0.9); margin-bottom: 15px;">
            <h3 style="margin: 0; color: #111111; font-weight: 900; font-size: 20px; text-align: center;">3.5 {page_mode} DWT — mô phỏng từng bước</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        clean_image = _load_clean_image()
    except FileNotFoundError:
        st.error("Không tìm thấy ảnh mẫu anh-trang-den-1.webp")
        return

    if mode == "setup":
        _render_setup(clean_image)
        return

    config = st.session_state.wavelet_config
    clean_image, noisy_image, stages = _load_forward_data(config)
    denoised_details, _, _ = _denoise_details(
        stages,
        config["method"],
        config["threshold_factor"],
    )

    if mode == "denoise":
        _render_denoise_demo(noisy_image, stages, config)
        return

    if mode == "inverse":
        _render_inverse_demo(noisy_image, stages, denoised_details)
        return

    if "forward_step" not in st.session_state:
        st.session_state.forward_step = 0
    step = st.session_state.forward_step

    control_left, control_mid, control_right = st.columns([1.2, 2.6, 1.2])
    with control_left:
        if st.button("↺ Bắt đầu lại", use_container_width=True):
            st.session_state.dwt_mode = "setup"
            st.rerun()
    with control_mid:
        st.progress(step / LEVELS)
        st.markdown(
            f"<div style='text-align:center; color:#4b5563;'>Bước {step}/{LEVELS}</div>",
            unsafe_allow_html=True,
        )
    with control_right:
        if st.button(
            "Next →" if step < LEVELS else "Đã hoàn tất",
            use_container_width=True,
            type="primary",
            disabled=step >= LEVELS,
        ):
            st.session_state.forward_step += 1
            st.rerun()

    if step < LEVELS:
        _render_process_area(stages[step])
    else:
        st.markdown("<div class='area-title'>Ô A — Forward DWT đã hoàn tất</div>", unsafe_allow_html=True)
        _image_card(stages[-1]["approximation"], "c₃(n) — nhánh xấp xỉ cuối", detail=False)
        if st.button("Tiếp tục lọc nhiễu Wavelet →", type="primary", use_container_width=True):
            st.session_state.dwt_mode = "denoise"
            st.rerun()

    stored = []
    for index in range(min(step, LEVELS)):
        stored.append(
            {
                "kind": "detail",
                "data": stages[index]["details"],
                "caption": f"d{index + 1}(n) — LH{index + 1} · HL{index + 1} · HH{index + 1}",
            }
        )
    if step >= LEVELS:
        stored.append({"kind": "approximation", "data": stages[-1]["approximation"], "caption": "c₃(n) — LL₃"})

    _render_storage_area(stored)
