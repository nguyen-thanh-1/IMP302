"""Interactive MMSE/Wiener filtering demonstration for section 3.6."""

import os

import cv2
import numpy as np
import streamlit as st


def _load_image():
    path = os.path.join(os.path.dirname(__file__), "anh-trang-den-1.webp")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if max(image.shape) > 512:
        scale = 512.0 / max(image.shape)
        image = cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image.astype(np.float32)


def _motion_psf(size, angle):
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(psf, (0, center), (size - 1, center), 1.0, 1)
    matrix = cv2.getRotationMatrix2D((center, center), float(angle), 1.0)
    psf = cv2.warpAffine(psf, matrix, (size, size), flags=cv2.INTER_CUBIC)
    psf = np.clip(psf, 0.0, None)
    return psf / max(float(psf.sum()), 1e-8)


def _gaussian_psf(size, sigma):
    kernel = cv2.getGaussianKernel(size, float(sigma))
    psf = (kernel @ kernel.T).astype(np.float32)
    return psf / max(float(psf.sum()), 1e-8)


def _psf_to_otf(psf, shape):
    padded = np.zeros(shape, dtype=np.float32)
    height, width = psf.shape
    padded[:height, :width] = psf
    padded = np.roll(padded, -(height // 2), axis=0)
    padded = np.roll(padded, -(width // 2), axis=1)
    return np.fft.fft2(padded)


def _crop(array, pad, shape):
    return array[pad:pad + shape[0], pad:pad + shape[1]]


def _build_degradation(clean, config):
    if config["blur_type"] == "Mờ chuyển động":
        psf = _motion_psf(config["kernel_size"], config["angle"])
    else:
        psf = _gaussian_psf(config["kernel_size"], config["blur_sigma"])

    pad = config["kernel_size"]
    clean_pad = np.pad(clean, ((pad, pad), (pad, pad)), mode="reflect")
    transfer = _psf_to_otf(psf, clean_pad.shape)
    clean_spectrum = np.fft.fft2(clean_pad)
    blurred_pad = np.real(np.fft.ifft2(clean_spectrum * transfer))

    rng = np.random.default_rng(int(config["seed"]))
    noise = rng.normal(0.0, float(config["noise_sigma"]), clean_pad.shape)
    degraded_pad = np.clip(blurred_pad + noise, 0.0, 255.0)

    return {
        "clean": clean,
        "clean_pad": clean_pad,
        "psf": psf,
        "transfer": transfer,
        "blurred": np.clip(_crop(blurred_pad, pad, clean.shape), 0, 255),
        "degraded": _crop(degraded_pad, pad, clean.shape),
        "degraded_pad": degraded_pad,
        "pad": pad,
    }


def _wiener_restore(data, noise_sigma, k_scale):
    clean_variance = max(float(np.var(data["clean_pad"])), 1e-8)
    noise_variance = float(noise_sigma) ** 2
    automatic_k = noise_variance / clean_variance
    k_value = max(automatic_k * float(k_scale), 1e-8)

    transfer = data["transfer"]
    wiener_transfer = np.conj(transfer) / (np.abs(transfer) ** 2 + k_value)
    degraded_spectrum = np.fft.fft2(data["degraded_pad"])
    restored_pad = np.real(np.fft.ifft2(wiener_transfer * degraded_spectrum))
    restored = _crop(restored_pad, data["pad"], data["clean"].shape)

    return np.clip(restored, 0, 255), wiener_transfer, automatic_k, k_value


def _to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def _spectrum_preview(spectrum):
    values = np.log1p(np.abs(np.fft.fftshift(spectrum))).astype(np.float32)
    low, high = np.percentile(values, (1, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _mse(reference, estimate):
    return float(np.mean((reference.astype(np.float32) - estimate.astype(np.float32)) ** 2))


def _snr(reference, estimate):
    signal_energy = max(float(np.sum(reference.astype(np.float64) ** 2)), 1e-12)
    error_energy = max(
        float(np.sum((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2)),
        1e-12,
    )
    return 10.0 * np.log10(signal_energy / error_energy)


def _title(text):
    st.markdown(
        f"<div class='wiener-title'>{text}</div>",
        unsafe_allow_html=True,
    )


def _render_setup(clean):
    _title("Thiết lập ảnh suy biến g(x,y)")
    left, right = st.columns(2)
    with left:
        blur_type = st.selectbox("Mô hình suy biến H", ["Mờ chuyển động", "Mờ Gaussian"])
        kernel_size = st.slider("Kích thước PSF", 5, 51, 21, step=2)
        if blur_type == "Mờ chuyển động":
            angle = st.slider("Góc chuyển động", 0, 180, 25, step=5)
            blur_sigma = 3.0
        else:
            blur_sigma = st.slider("Độ lệch chuẩn mờ Gaussian", 0.5, 10.0, 3.0, step=0.5)
            angle = 0
    with right:
        noise_sigma = st.slider("Nhiễu Gaussian σₙ", 0.0, 40.0, 10.0, step=1.0)
        seed = st.number_input("Seed nhiễu", min_value=0, max_value=9999, value=42, step=1)

    config = {
        "blur_type": blur_type,
        "kernel_size": kernel_size,
        "angle": angle,
        "blur_sigma": blur_sigma,
        "noise_sigma": noise_sigma,
        "seed": int(seed),
    }
    preview = _build_degradation(clean, config)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(_to_uint8(clean), caption="Ảnh sạch f(x,y)", channels="GRAY", width="stretch")
    with col2:
        st.image(_to_uint8(preview["blurred"]), caption="Hf — ảnh bị mờ", channels="GRAY", width="stretch")
    with col3:
        st.image(_to_uint8(preview["degraded"]), caption="g = Hf + η", channels="GRAY", width="stretch")

    if st.button("Bắt đầu mô phỏng Wiener →", type="primary", width="stretch"):
        st.session_state.wiener_config = config
        st.session_state.wiener_mode = "demo"
        st.session_state.wiener_step = 0
        st.session_state.wiener_k_scale = 1.0
        st.rerun()


def _render_degradation(data):
    _title("Bước 1 — Tạo ảnh suy biến")
    columns = st.columns([1.0, 0.18, 1.0, 0.18, 1.0])
    with columns[0]:
        st.image(_to_uint8(data["clean"]), caption="f(x,y)", channels="GRAY", width="stretch")
    with columns[1]:
        st.markdown("<div class='wiener-arrow'>→ H →</div>", unsafe_allow_html=True)
    with columns[2]:
        st.image(_to_uint8(data["blurred"]), caption="Hf", channels="GRAY", width="stretch")
    with columns[3]:
        st.markdown("<div class='wiener-arrow'>+ η →</div>", unsafe_allow_html=True)
    with columns[4]:
        st.image(_to_uint8(data["degraded"]), caption="g(x,y)", channels="GRAY", width="stretch")


def _render_frequency(data):
    noise_sigma = st.session_state.wiener_config["noise_sigma"]
    _, wiener_transfer, automatic_k, k_value = _wiener_restore(data, noise_sigma, 1.0)
    _title("Bước 2 — Bộ lọc Wiener trong miền tần số")
    st.latex(
        r"\hat F(u,v)=\left[\frac{1}{H(u,v)}\frac{|H(u,v)|^2}{|H(u,v)|^2+K}\right]G(u,v),"
        r"\qquad K\approx\frac{S_\eta(u,v)}{S_f(u,v)}"
    )
    degraded_spectrum = np.fft.fft2(data["degraded_pad"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(_spectrum_preview(degraded_spectrum), caption="|G(u,v)| — phổ ảnh suy biến", channels="GRAY", width="stretch")
    with col2:
        st.image(_spectrum_preview(data["transfer"]), caption="|H(u,v)| — hàm truyền", channels="GRAY", width="stretch")
    with col3:
        st.image(_spectrum_preview(wiener_transfer), caption="|W(u,v)| — bộ lọc Wiener", channels="GRAY", width="stretch")
    st.markdown(f"<div class='wiener-k'>K tự động = {automatic_k:.5f} · K đang dùng = {k_value:.5f}</div>", unsafe_allow_html=True)


def _render_result(data):
    _title("Bước 3 — Tối thiểu hóa sai số bình phương trung bình")
    st.latex(r"\hat f=\arg\min_{\tilde f}\;\mathbb{E}\left\{(f-\tilde f)^2\right\}")
    formula_left, formula_right = st.columns(2)
    with formula_left:
        st.latex(r"\mathrm{MSE}=\frac{1}{MN}\sum_x\sum_y[f(x,y)-\hat f(x,y)]^2")
    with formula_right:
        st.latex(r"\mathrm{SNR}_{dB}=10\log_{10}\frac{\sum_x\sum_y f(x,y)^2}{\sum_x\sum_y[f(x,y)-\hat f(x,y)]^2}")
    k_scale = st.slider(
        "Hệ số nhân cho K = Sη/Sf",
        0.05,
        5.0,
        value=float(st.session_state.get("wiener_k_scale", 1.0)),
        step=0.05,
        key="wiener_k_scale",
    )
    noise_sigma = st.session_state.wiener_config["noise_sigma"]
    restored, _, automatic_k, k_value = _wiener_restore(data, noise_sigma, k_scale)

    degraded_mse = _mse(data["clean"], data["degraded"])
    restored_mse = _mse(data["clean"], restored)
    degraded_snr = _snr(data["clean"], data["degraded"])
    restored_snr = _snr(data["clean"], restored)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(_to_uint8(data["clean"]), caption="Ảnh sạch f", channels="GRAY", width="stretch")
    with col2:
        st.image(_to_uint8(data["degraded"]), caption=f"Ảnh suy biến g · MSE {degraded_mse:.1f}", channels="GRAY", width="stretch")
    with col3:
        st.image(_to_uint8(restored), caption=f"Ảnh Wiener f̂ · MSE {restored_mse:.1f}", channels="GRAY", width="stretch")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("MSE trước", f"{degraded_mse:.2f}")
    metric2.metric("MSE sau", f"{restored_mse:.2f}", delta=f"{restored_mse - degraded_mse:.2f}", delta_color="inverse")
    metric3.metric("ΔMSE (trước − sau)", f"{degraded_mse - restored_mse:+.2f}")

    metric4, metric5, metric6 = st.columns(3)
    metric4.metric("SNR trước", f"{degraded_snr:.2f} dB")
    metric5.metric("SNR sau", f"{restored_snr:.2f} dB", delta=f"{restored_snr - degraded_snr:+.2f} dB")
    metric6.metric("ΔSNR (sau − trước)", f"{restored_snr - degraded_snr:+.2f} dB")
    st.markdown(
        f"<div class='wiener-k'>K tự động = {automatic_k:.5f} · hệ số nhân = {k_scale:.2f} · {automatic_k:.5f} = {k_value:.5f}</div>",
        unsafe_allow_html=True,
    )


def render_wiener_demo():
    st.markdown(
        """
        <style>
        .wiener-header, .wiener-title {
            background: #e0e5ec; border-radius: 15px; text-align: center; color: #111827;
            font-weight: 900; box-shadow: 6px 6px 12px rgba(163,177,198,.55), -6px -6px 12px rgba(255,255,255,.9);
        }
        .wiener-header { padding: 15px; margin-bottom: 16px; font-size: 21px; }
        .wiener-title { padding: 13px 16px; margin: 10px 0 16px; font-size: 20px; }
        .wiener-arrow { min-height: 250px; display: flex; align-items: center; justify-content: center; color: #7058e8; font-size: 17px; font-weight: 900; white-space: nowrap; }
        .wiener-k { text-align: center; color: #4b5563; font-weight: 700; margin: 10px 0 4px; }
        @media (max-width: 900px) { .wiener-arrow { min-height: 150px; font-size: 13px; } }
        </style>
        <div class='wiener-header'>3.6 Minimum Mean Square Error Filtering — Wiener Filter</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        clean = _load_image()
    except FileNotFoundError:
        st.error("Không tìm thấy ảnh mẫu anh-trang-den-1.webp")
        return

    if "wiener_mode" not in st.session_state:
        st.session_state.wiener_mode = "setup"
    if st.session_state.wiener_mode == "setup" or "wiener_config" not in st.session_state:
        _render_setup(clean)
        return

    data = _build_degradation(clean, st.session_state.wiener_config)
    step = int(st.session_state.get("wiener_step", 0))
    left, middle, right = st.columns([1.2, 2.6, 1.2])
    with left:
        if st.button("↺ Thiết lập lại", width="stretch"):
            st.session_state.wiener_mode = "setup"
            st.rerun()
    with middle:
        st.progress((step + 1) / 3.0)
        st.markdown(f"<div class='wiener-k'>Bước {step + 1}/3</div>", unsafe_allow_html=True)
    with right:
        if st.button(
            "Next →" if step < 2 else "Đã hoàn tất",
            type="primary",
            disabled=step >= 2,
            width="stretch",
        ):
            st.session_state.wiener_step += 1
            st.rerun()

    if step == 0:
        _render_degradation(data)
    elif step == 1:
        _render_frequency(data)
    else:
        _render_result(data)
