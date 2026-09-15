#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 体电镜图像Z轴条纹数据生成与预处理
# 支持五种条纹模拟方式随机混合：
# 1) feijunyun: 非均匀场 + Z向亮度/对比度漂移（已抑制过强黑白饱和）
# 2) noise: 基础亮度/对比度变化 + EM噪声（已抑制过强黑白饱和）
# 3) basic: 基础亮度/对比度变化（已抑制过强黑白饱和）
# 4) rect_block: 随机大片矩形/斜矩形区域亮度提高、对比度下降（重叠不再叠加发白）
# 5) full_slice: 整片亮度提高、对比度下降

import os
import argparse
import numpy as np
from tqdm import tqdm
import tifffile as tiff
from PIL import Image


# ======================
# 配置参数（可修改）
# ======================

VAL_RATIO = 0.1
TEST_RATIO = 0.1

USE_FIXED_SEED = True
SEED = 42

IMAGE_SIZE = 1024
OUTPUT_FORMAT = 'png'

MIX_STRIPE_MODES = True

# 顺序：feijunyun, noise, basic, rect_block, full_slice
MODE_WEIGHTS = [0.30, 0.15, 0.15, 0.15, 0.25]


# ======================
# 通用条纹参数（原 feijunyun）
# ======================

STRENGTH = 1.0

DOMINANT_PERIOD = 7.5
PERIOD_JITTER = 0.16

ALPHA_STD = 0.040 * STRENGTH
BETA_STD = 0.028 * STRENGTH

ALPHA_LIMIT = 0.12 * STRENGTH
BETA_LIMIT = 0.10 * STRENGTH

ALPHA_BETA_CORR = 0.92

ENVELOPE_STD = 0.65
ENVELOPE_SMOOTH = 28

BURST_PROB = 0.012
BURST_LEN_RANGE = (6, 20)
BURST_BETA_AMP = 0.10 * STRENGTH
BURST_ALPHA_AMP = 0.05 * STRENGTH

SHARPEN_WEIGHT = 0.55
HF_NOISE_STD = 0.30

USE_HEAVY_TAIL_BETA = True
T_DF = 3.0
HEAVY_TAIL_MIX = 0.35

ENABLE_SPATIAL_FIELDS = True

SPATIAL_ALPHA_STD = 0.030 * STRENGTH
SPATIAL_BETA_STD = 0.035 * STRENGTH

SPATIAL_ALPHA_BETA_CORR = 0.85

W_SHADING = 0.60
W_PATCH = 0.25
W_SCANLINE = 0.15

SHADING_SCALE = 96
SHADING_SMOOTH = 81

PATCH_SCALE = 24
PATCH_SMOOTH = 25

SCANLINE_SMOOTH = 17

DRIFT_PX_STD = 0.35
DRIFT_SMOOTH_Z = 35

REFRESH_STD = 0.15
REFRESH_SMOOTH = 11

USE_LOCAL_BASELINE = True
BASELINE_SMOOTH = 81


# ======================
# 噪声模式参数
# ======================

USE_MED_CENTERED_CONTRAST = True
ADD_NOISE = True
GAUSS_NOISE_STD = 2.0 * STRENGTH
USE_POISSON = True
POISSON_SCALE = 40.0 / STRENGTH
CORR_NOISE_STD = 1.2 * STRENGTH
CORR_NOISE_SMOOTH = 2


# ======================
# 前三种模式的大范围随机亮度/对比度扰动（已收窄）
# ======================

AFFINE_STRENGTH_PROBS = [0.50, 0.35, 0.15]

# 更稳妥的对比度范围，避免极端黑白
AFFINE_ALPHA_SMALL = (-0.05, 0.05)
AFFINE_ALPHA_MEDIUM = (-0.12, 0.12)
AFFINE_ALPHA_LARGE = (-0.22, 0.22)

# 更稳妥的亮度范围，避免强烈漂移到纯黑纯白
AFFINE_BETA_SMALL = (-0.03, 0.03)
AFFINE_BETA_MEDIUM = (-0.08, 0.08)
AFFINE_BETA_LARGE = (-0.15, 0.15)

# 总对比度限制
GLOBAL_ALPHA_MIN = 0.60
GLOBAL_ALPHA_MAX = 1.40

FEIJUNYUN_EXTRA_ALPHA_SCALE = 1.00
FEIJUNYUN_EXTRA_BETA_SCALE = 1.00

NOISE_EXTRA_ALPHA_SCALE = 1.00
NOISE_EXTRA_BETA_SCALE = 1.00

BASIC_EXTRA_ALPHA_SCALE = 1.00
BASIC_EXTRA_BETA_SCALE = 1.00

# 前三种模式最终亮度偏移限制（相对 max_val_for_beta）
FINAL_BETA_CLIP_RATIO = 0.22

# 软限制参数：在输入图像分位区间外平滑压缩，避免纯黑纯白
SOFT_CLIP_LOW_Q = 0.003
SOFT_CLIP_HIGH_Q = 0.997
SOFT_CLIP_EXPAND = 0.18


# ======================
# 局部块状亮度升高 / 对比度降低模式参数
# ======================

RECT_MODE_ALPHA_RANGE = (0.68, 0.90)
RECT_MODE_BETA_RANGE = (0.05, 0.14)

RECT_NUM_RANGE = (1, 3)
RECT_W_RATIO_RANGE = (0.30, 0.85)
RECT_H_RATIO_RANGE = (0.20, 0.70)
RECT_ANGLE_RANGE = (-35.0, 35.0)
RECT_EDGE_BLUR = 9


# ======================
# 整片亮度升高 / 对比度降低模式参数
# ======================

FULL_MODE_ALPHA_RANGE = (0.72, 0.92)
FULL_MODE_BETA_RANGE = (0.04, 0.12)
FULL_USE_MEDIAN_CENTER = True


# ======================
# 固定范围保存参数
# ======================

USE_FIXED_SAVE_RANGE = True
SAVE_RANGE_LOW_PERCENTILE = 1.0
SAVE_RANGE_HIGH_PERCENTILE = 99.0


# ======================
# 工具函数
# ======================

def _get_max_val(dtype: np.dtype) -> float:
    if dtype == np.uint8:
        return 255.0
    if dtype == np.uint16:
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return 1.0
    raise TypeError(f"Unsupported dtype: {dtype}")


def moving_average_1d(x: np.ndarray, w: int) -> np.ndarray:
    x = x.astype(np.float32)
    n = len(x)
    if w <= 1 or n == 0:
        return x
    w = min(int(w), n)
    if w <= 1:
        return x
    k = np.ones(w, dtype=np.float32) / float(w)
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    xp = np.pad(x, (pad_left, pad_right), mode="reflect")
    y = np.convolve(xp, k, mode="valid")
    return y.astype(np.float32)


def normalize_to_std(x: np.ndarray, target_std: float) -> np.ndarray:
    x = x.astype(np.float32)
    s = float(x.std() + 1e-8)
    return (x / s * float(target_std)).astype(np.float32)


def zero_mean_unit_std(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.mean())
    x = x / float(x.std() + 1e-8)
    return x


def quasi_periodic_oscillator(z: int, rng: np.random.Generator, period: float, jitter: float) -> np.ndarray:
    base_w = 2.0 * np.pi / float(period)
    eps = rng.normal(0.0, jitter, size=z).astype(np.float32)
    eps = moving_average_1d(eps, 9)
    w = base_w * (1.0 + eps)
    phase = np.cumsum(w).astype(np.float32)
    phase += float(rng.uniform(0, 2 * np.pi))
    s = np.sin(phase).astype(np.float32)
    s -= s.mean()
    return s


def make_envelope(z: int, rng: np.random.Generator, std: float, smooth: int) -> np.ndarray:
    a = rng.normal(0.0, 1.0, size=z).astype(np.float32)
    a = moving_average_1d(a, smooth)
    a = normalize_to_std(a, std)
    env = 1.0 + a
    return np.clip(env, 0.3, 2.5).astype(np.float32)


def second_diff_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = len(x)
    y = np.zeros_like(x)

    if n == 1:
        return y
    if n == 2:
        y[0] = x[1] - x[0]
        y[1] = x[1] - x[0]
        return y

    y[1:-1] = x[2:] - 2 * x[1:-1] + x[:-2]
    y[0] = y[1]
    y[-1] = y[-2]
    y -= y.mean()
    return y


def apply_quantile_clip(x: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    lo = float(np.quantile(x, low_q))
    hi = float(np.quantile(x, high_q))
    if hi <= lo:
        return x.astype(np.float32)
    return np.clip(x, lo, hi).astype(np.float32)


def heavy_tail_noise(z: int, rng: np.random.Generator, df: float) -> np.ndarray:
    x = rng.standard_t(df, size=z).astype(np.float32)
    x -= x.mean()
    return x


def add_bursts(alpha_delta: np.ndarray, beta_delta: np.ndarray, rng: np.random.Generator):
    z = len(alpha_delta)
    i = 0
    while i < z:
        if rng.random() < BURST_PROB:
            L = int(rng.integers(BURST_LEN_RANGE[0], BURST_LEN_RANGE[1] + 1))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            t = rng.normal(0.0, 1.0, size=L).astype(np.float32)
            t = moving_average_1d(t, 3)
            t = t / (t.std() + 1e-8)
            for k in range(L):
                j = i + k
                if j >= z:
                    break
                beta_delta[j] += sign * BURST_BETA_AMP * (0.7 + 0.3 * t[k])
                alpha_delta[j] += sign * BURST_ALPHA_AMP * (0.7 + 0.3 * t[k])
            i += L
        else:
            i += 1
    return alpha_delta, beta_delta


def box_filter_1d_reflect(a: np.ndarray, w: int, axis: int) -> np.ndarray:
    a = a.astype(np.float32)
    if w <= 1:
        return a
    w = int(w)
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad_left, pad_right)
    ap = np.pad(a, pad_width, mode="reflect")

    cp = np.cumsum(ap, axis=axis, dtype=np.float32)
    zshape = list(cp.shape)
    zshape[axis] = 1
    zeros = np.zeros(zshape, dtype=np.float32)
    cp0 = np.concatenate([zeros, cp], axis=axis)

    n = a.shape[axis]
    sl_end = [slice(None)] * a.ndim
    sl_sta = [slice(None)] * a.ndim
    sl_end[axis] = slice(w, w + n)
    sl_sta[axis] = slice(0, n)
    s = cp0[tuple(sl_end)] - cp0[tuple(sl_sta)]
    return (s / float(w)).astype(np.float32)


def box_blur_2d_reflect(img: np.ndarray, wy: int, wx: int) -> np.ndarray:
    out = box_filter_1d_reflect(img, wy, axis=0)
    out = box_filter_1d_reflect(out, wx, axis=1)
    return out.astype(np.float32)


def make_lowfreq_field(y: int, x: int, rng: np.random.Generator, scale: int, smooth: int) -> np.ndarray:
    gy = max(2, int(np.round(y / float(scale))))
    gx = max(2, int(np.round(x / float(scale))))
    g = rng.normal(0.0, 1.0, size=(gy, gx)).astype(np.float32)

    fy = int(np.ceil(y / gy))
    fx = int(np.ceil(x / gx))
    up = np.repeat(np.repeat(g, fy, axis=0), fx, axis=1)[:y, :x]

    up = box_blur_2d_reflect(up, smooth, smooth)
    return zero_mean_unit_std(up)


def make_scanline_field(y: int, x: int, rng: np.random.Generator, smooth: int, direction: str = "y") -> np.ndarray:
    if direction == "y":
        v = rng.normal(0.0, 1.0, size=y).astype(np.float32)
        v = moving_average_1d(v, smooth)
        v = zero_mean_unit_std(v)
        field = v[:, None] * np.ones((1, x), dtype=np.float32)
    else:
        v = rng.normal(0.0, 1.0, size=x).astype(np.float32)
        v = moving_average_1d(v, smooth)
        v = zero_mean_unit_std(v)
        field = np.ones((y, 1), dtype=np.float32) * v[None, :]
    return field.astype(np.float32)


def smooth2d_box(img: np.ndarray, w: int) -> np.ndarray:
    img = img.astype(np.float32)
    if w <= 1:
        return img

    w = int(w)
    k = np.ones(w, dtype=np.float32) / float(w)

    pad_left = w // 2
    pad_right = w - 1 - pad_left

    tmp = np.pad(img, ((pad_left, pad_right), (0, 0)), mode="reflect")
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, tmp)

    tmp2 = np.pad(tmp, ((0, 0), (pad_left, pad_right)), mode="reflect")
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, tmp2)

    return out.astype(np.float32)


def soft_clip_to_valid_range(sim: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    """
    参考原图动态范围做软限制，避免大量像素直接撞到纯黑/纯白。
    """
    ref = ref_img.astype(np.float32)
    sim = sim.astype(np.float32)

    lo = float(np.quantile(ref, SOFT_CLIP_LOW_Q))
    hi = float(np.quantile(ref, SOFT_CLIP_HIGH_Q))
    if hi <= lo:
        return sim

    span = hi - lo
    lo2 = lo - SOFT_CLIP_EXPAND * span
    hi2 = hi + SOFT_CLIP_EXPAND * span
    center = 0.5 * (lo2 + hi2)
    half = 0.5 * (hi2 - lo2)

    if half <= 1e-8:
        return sim

    y = (sim - center) / half
    y = np.tanh(y)
    out = center + half * y
    return out.astype(np.float32)


def generate_z_profiles(z: int, rng: np.random.Generator):
    base = quasi_periodic_oscillator(z, rng, DOMINANT_PERIOD, PERIOD_JITTER)
    base = normalize_to_std(base, 1.0)

    env = make_envelope(z, rng, ENVELOPE_STD, ENVELOPE_SMOOTH)

    hf = rng.normal(0.0, 1.0, size=z).astype(np.float32)
    hf = second_diff_1d(moving_average_1d(hf, 2))
    hf = normalize_to_std(hf, 1.0)

    base2 = (1.0 - HF_NOISE_STD) * base + HF_NOISE_STD * hf
    base2 = base2 * env
    base2 -= base2.mean()

    sharp = second_diff_1d(base2)
    sharp = normalize_to_std(sharp, 1.0)
    src = (1.0 - SHARPEN_WEIGHT) * base2 + SHARPEN_WEIGHT * sharp
    src -= src.mean()
    src = normalize_to_std(src, 1.0)

    alpha_delta = normalize_to_std(src, ALPHA_STD)

    indep = rng.normal(0.0, 1.0, size=z).astype(np.float32)
    indep = moving_average_1d(indep, 3)
    indep -= indep.mean()
    indep = normalize_to_std(indep, 1.0)

    beta_src = ALPHA_BETA_CORR * normalize_to_std(alpha_delta, 1.0) + \
        np.sqrt(max(1e-6, 1.0 - ALPHA_BETA_CORR ** 2)) * indep

    if USE_HEAVY_TAIL_BETA:
        ht = heavy_tail_noise(z, rng, T_DF)
        ht = normalize_to_std(ht, 1.0)
        beta_src = (1.0 - HEAVY_TAIL_MIX) * beta_src + HEAVY_TAIL_MIX * ht

    beta_delta = normalize_to_std(beta_src * env, BETA_STD)

    alpha_delta, beta_delta = add_bursts(alpha_delta, beta_delta, rng)

    alpha_delta = apply_quantile_clip(alpha_delta, 0.002, 0.998)
    beta_delta = apply_quantile_clip(beta_delta, 0.002, 0.998)

    alpha_delta = np.clip(alpha_delta, -ALPHA_LIMIT, ALPHA_LIMIT).astype(np.float32)
    beta_delta = np.clip(beta_delta, -BETA_LIMIT, BETA_LIMIT).astype(np.float32)

    alpha = np.clip(1.0 + alpha_delta, 1.0 - ALPHA_LIMIT, 1.0 + ALPHA_LIMIT).astype(np.float32)
    beta_ratio = beta_delta.astype(np.float32)

    return alpha, beta_ratio, env.astype(np.float32)


def prepare_spatial_patterns(y: int, x: int, rng: np.random.Generator):
    shading0 = make_lowfreq_field(y, x, rng, SHADING_SCALE, SHADING_SMOOTH)
    patch0 = make_lowfreq_field(y, x, rng, PATCH_SCALE, PATCH_SMOOTH)
    line0 = make_scanline_field(y, x, rng, SCANLINE_SMOOTH, direction="y")

    base = (W_SHADING * shading0 + W_PATCH * patch0 + W_SCANLINE * line0).astype(np.float32)
    base = zero_mean_unit_std(base)
    return base


def prepare_drift_offsets(z: int, rng: np.random.Generator):
    step_y = rng.normal(0.0, DRIFT_PX_STD, size=z).astype(np.float32)
    step_x = rng.normal(0.0, DRIFT_PX_STD, size=z).astype(np.float32)
    step_y = moving_average_1d(step_y, DRIFT_SMOOTH_Z)
    step_x = moving_average_1d(step_x, DRIFT_SMOOTH_Z)
    off_y = np.round(np.cumsum(step_y)).astype(np.int32)
    off_x = np.round(np.cumsum(step_x)).astype(np.int32)
    return off_y, off_x


def make_spatial_maps_for_slice(i: int, base_pattern: np.ndarray, rng: np.random.Generator,
                                off_y: np.ndarray, off_x: np.ndarray):
    y, x = base_pattern.shape
    pat = np.roll(base_pattern, shift=(int(off_y[i]), int(off_x[i])), axis=(0, 1)).astype(np.float32)

    if REFRESH_STD > 1e-6:
        n = rng.normal(0.0, 1.0, size=(y, x)).astype(np.float32)
        n = box_blur_2d_reflect(n, REFRESH_SMOOTH, REFRESH_SMOOTH)
        n = zero_mean_unit_std(n)
        pat = (1.0 - REFRESH_STD) * pat + REFRESH_STD * n
        pat = zero_mean_unit_std(pat)

    delta_a = pat

    indep = rng.normal(0.0, 1.0, size=(y, x)).astype(np.float32)
    indep = box_blur_2d_reflect(indep, 9, 9)
    indep = zero_mean_unit_std(indep)

    c = float(np.clip(SPATIAL_ALPHA_BETA_CORR, -0.999, 0.999))
    delta_b = c * delta_a + np.sqrt(max(1e-6, 1.0 - c * c)) * indep
    delta_b = zero_mean_unit_std(delta_b)

    return delta_a.astype(np.float32), delta_b.astype(np.float32)


def add_em_noise(sim: np.ndarray, rng: np.random.Generator, max_val: float) -> np.ndarray:
    out = sim.astype(np.float32)

    if USE_POISSON:
        s = float(max(1e-6, POISSON_SCALE))
        lam = np.clip(out / max_val, 0.0, 1.0) * s
        noisy = rng.poisson(lam).astype(np.float32) / s
        out = noisy * max_val

    if GAUSS_NOISE_STD > 0:
        out = out + rng.normal(0.0, GAUSS_NOISE_STD, size=out.shape).astype(np.float32)

    if CORR_NOISE_STD > 0:
        n = rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
        n = smooth2d_box(n, CORR_NOISE_SMOOTH)
        n = n / (n.std() + 1e-8)
        out = out + n * float(CORR_NOISE_STD)

    return out.astype(np.float32)


def clip_by_dtype(img: np.ndarray, dtype, max_val_for_beta: float) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return img.astype(dtype)
    return np.clip(img, 0.0, max_val_for_beta).astype(np.float32)


def affine_contrast_brightness(img: np.ndarray, alpha: float, beta: float, center: float = None) -> np.ndarray:
    img = img.astype(np.float32)
    if center is None:
        center = float(np.median(img))
    return center + alpha * (img - center) + beta


def make_rotated_rectangle_mask(h: int, w: int,
                                cx: float, cy: float,
                                rw: float, rh: float,
                                angle_deg: float,
                                blur_w: int = 9) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    x = xx - cx
    y = yy - cy

    theta = np.deg2rad(angle_deg)
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))

    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t

    mask = ((np.abs(xr) <= rw / 2.0) & (np.abs(yr) <= rh / 2.0)).astype(np.float32)

    if blur_w > 1:
        mask = smooth2d_box(mask, blur_w)
        mask = np.clip(mask, 0.0, 1.0)

    return mask.astype(np.float32)


# ======================
# 前三种模式的随机亮度/对比度采样
# ======================

def sample_mixed_range(rng: np.random.Generator,
                       probs,
                       small_range,
                       medium_range,
                       large_range) -> float:
    k = int(rng.choice(3, p=np.array(probs, dtype=np.float64) / np.sum(probs)))
    if k == 0:
        lo, hi = small_range
    elif k == 1:
        lo, hi = medium_range
    else:
        lo, hi = large_range
    return float(rng.uniform(lo, hi))


def sample_extra_affine(rng: np.random.Generator, max_val_for_beta: float):
    alpha_delta = sample_mixed_range(
        rng,
        AFFINE_STRENGTH_PROBS,
        AFFINE_ALPHA_SMALL,
        AFFINE_ALPHA_MEDIUM,
        AFFINE_ALPHA_LARGE
    )
    alpha_extra = 1.0 + alpha_delta
    alpha_extra = float(np.clip(alpha_extra, GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX))

    beta_ratio = sample_mixed_range(
        rng,
        AFFINE_STRENGTH_PROBS,
        AFFINE_BETA_SMALL,
        AFFINE_BETA_MEDIUM,
        AFFINE_BETA_LARGE
    )
    beta_extra = float(beta_ratio * max_val_for_beta)

    beta_limit = FINAL_BETA_CLIP_RATIO * max_val_for_beta
    beta_extra = float(np.clip(beta_extra, -beta_limit, beta_limit))

    return alpha_extra, beta_extra


# ======================
# 五种条纹模拟模式
# ======================

def apply_feijunyun_mode(img: np.ndarray, alpha_z: np.ndarray, beta_z_global: np.ndarray,
                         env_z: np.ndarray, base_pattern, off_y, off_x, i: int,
                         rng: np.random.Generator, max_val_for_beta: float, dtype):
    img = img.astype(np.float32)

    if USE_LOCAL_BASELINE:
        base = box_blur_2d_reflect(img, BASELINE_SMOOTH, BASELINE_SMOOTH)
    else:
        base = float(np.median(img))

    a0 = float(alpha_z[i])
    b0 = float(beta_z_global[i])

    alpha_extra, beta_extra = sample_extra_affine(rng, max_val_for_beta)
    a0 = float(np.clip(a0 * (1.0 + FEIJUNYUN_EXTRA_ALPHA_SCALE * (alpha_extra - 1.0)),
                       GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX))
    b0 = float(b0 + FEIJUNYUN_EXTRA_BETA_SCALE * beta_extra)

    beta_limit = FINAL_BETA_CLIP_RATIO * max_val_for_beta
    b0 = float(np.clip(b0, -beta_limit, beta_limit))

    if ENABLE_SPATIAL_FIELDS:
        delta_a, delta_b = make_spatial_maps_for_slice(i, base_pattern, rng, off_y, off_x)
        env = float(env_z[i])

        alpha_map = a0 + (SPATIAL_ALPHA_STD * env) * delta_a
        beta_map = b0 + (SPATIAL_BETA_STD * env) * delta_b * max_val_for_beta

        alpha_map = np.clip(alpha_map, GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX).astype(np.float32)
        beta_map = np.clip(beta_map, -beta_limit, beta_limit).astype(np.float32)

        sim = base + alpha_map * (img - base) + beta_map
    else:
        sim = base + a0 * (img - base) + b0

    sim = soft_clip_to_valid_range(sim, img)
    sim = clip_by_dtype(sim, dtype, max_val_for_beta)
    return sim


def apply_noise_mode(img: np.ndarray, alpha_z: np.ndarray, beta_z_global: np.ndarray,
                     rng: np.random.Generator, max_val_for_beta: float, dtype):
    img = img.astype(np.float32)
    a = float(alpha_z[len(alpha_z) // 2])
    b = float(beta_z_global[len(beta_z_global) // 2])

    alpha_extra, beta_extra = sample_extra_affine(rng, max_val_for_beta)
    a = float(np.clip(a * (1.0 + NOISE_EXTRA_ALPHA_SCALE * (alpha_extra - 1.0)),
                      GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX))
    b = float(b + NOISE_EXTRA_BETA_SCALE * beta_extra)

    beta_limit = FINAL_BETA_CLIP_RATIO * max_val_for_beta
    b = float(np.clip(b, -beta_limit, beta_limit))

    if USE_MED_CENTERED_CONTRAST:
        med = float(np.median(img))
        sim = med + a * (img - med) + b
    else:
        sim = a * img + b

    sim = soft_clip_to_valid_range(sim, img)

    if ADD_NOISE:
        sim = add_em_noise(sim, rng, max_val_for_beta)
        sim = soft_clip_to_valid_range(sim, img)

    sim = clip_by_dtype(sim, dtype, max_val_for_beta)
    return sim


def apply_basic_mode(img: np.ndarray, alpha_z: np.ndarray, beta_z_global: np.ndarray,
                     rng: np.random.Generator, max_val_for_beta: float, dtype):
    img = img.astype(np.float32)
    a = float(alpha_z[len(alpha_z) // 2])
    b = float(beta_z_global[len(beta_z_global) // 2])

    alpha_extra, beta_extra = sample_extra_affine(rng, max_val_for_beta)
    a = float(np.clip(a * (1.0 + BASIC_EXTRA_ALPHA_SCALE * (alpha_extra - 1.0)),
                      GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX))
    b = float(b + BASIC_EXTRA_BETA_SCALE * beta_extra)

    beta_limit = FINAL_BETA_CLIP_RATIO * max_val_for_beta
    b = float(np.clip(b, -beta_limit, beta_limit))

    if USE_MED_CENTERED_CONTRAST:
        med = float(np.median(img))
        sim = med + a * (img - med) + b
    else:
        sim = a * img + b

    sim = soft_clip_to_valid_range(sim, img)
    sim = clip_by_dtype(sim, dtype, max_val_for_beta)
    return sim


def apply_rect_bright_lowcontrast_mode(img: np.ndarray,
                                       rng: np.random.Generator,
                                       max_val_for_beta: float,
                                       dtype):
    """
    改进版：
    1. 同一张图所有块共享同一 alpha/beta
    2. 先合并成 union mask
    3. 最后只做一次变换
    这样重叠区域不会叠加发白
    """
    img = img.astype(np.float32)
    h, w = img.shape

    num_rect = int(rng.integers(RECT_NUM_RANGE[0], RECT_NUM_RANGE[1] + 1))

    # 同一张图共享同一组块参数，避免重叠叠加变得更白
    alpha = float(rng.uniform(RECT_MODE_ALPHA_RANGE[0], RECT_MODE_ALPHA_RANGE[1]))
    beta = float(rng.uniform(RECT_MODE_BETA_RANGE[0], RECT_MODE_BETA_RANGE[1])) * max_val_for_beta

    union_mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_rect):
        rw = float(rng.uniform(RECT_W_RATIO_RANGE[0], RECT_W_RATIO_RANGE[1]) * w)
        rh = float(rng.uniform(RECT_H_RATIO_RANGE[0], RECT_H_RATIO_RANGE[1]) * h)

        cx = float(rng.uniform(0.10 * w, 0.90 * w))
        cy = float(rng.uniform(0.10 * h, 0.90 * h))
        angle = float(rng.uniform(RECT_ANGLE_RANGE[0], RECT_ANGLE_RANGE[1]))

        mask = make_rotated_rectangle_mask(
            h=h, w=w,
            cx=cx, cy=cy,
            rw=rw, rh=rh,
            angle_deg=angle,
            blur_w=RECT_EDGE_BLUR
        )

        # 使用并集，不叠加强度
        union_mask = np.maximum(union_mask, mask)

    center = float(np.median(img))
    transformed = affine_contrast_brightness(img, alpha=alpha, beta=beta, center=center)

    sim = (1.0 - union_mask) * img + union_mask * transformed
    sim = soft_clip_to_valid_range(sim, img)
    sim = clip_by_dtype(sim, dtype, max_val_for_beta)
    return sim


def apply_full_bright_lowcontrast_mode(img: np.ndarray,
                                       rng: np.random.Generator,
                                       max_val_for_beta: float,
                                       dtype):
    img = img.astype(np.float32)

    alpha = float(rng.uniform(FULL_MODE_ALPHA_RANGE[0], FULL_MODE_ALPHA_RANGE[1]))
    beta = float(rng.uniform(FULL_MODE_BETA_RANGE[0], FULL_MODE_BETA_RANGE[1])) * max_val_for_beta

    if FULL_USE_MEDIAN_CENTER:
        center = float(np.median(img))
    else:
        center = float(img.mean())

    sim = affine_contrast_brightness(img, alpha=alpha, beta=beta, center=center)
    sim = soft_clip_to_valid_range(sim, img)
    sim = clip_by_dtype(sim, dtype, max_val_for_beta)
    return sim


# ======================
# 主处理函数
# ======================

def split_indices(total: int, val_ratio: float, test_ratio: float, seed: int):
    np.random.seed(seed)
    indices = np.random.permutation(total)

    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    return train_indices, val_indices, test_indices


def save_image(img: np.ndarray, path: str, image_size: int = None,
               save_lo: float = None, save_hi: float = None):
    img = img.astype(np.float32)

    if save_lo is not None and save_hi is not None and save_hi > save_lo:
        img = np.clip((img - save_lo) / (save_hi - save_lo + 1e-8), 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    else:
        img_min, img_max = float(img.min()), float(img.max())
        if img_max > img_min:
            if img_max > 255.0 or img_min < 0.0:
                img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)

    if image_size is not None:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
        img = np.array(pil_img)

    pil_img = Image.fromarray(img)
    pil_img.save(path)


def process_volume(input_path: str, output_dir: str, split: str,
                   val_ratio: float, test_ratio: float, max_samples: int,
                   seed: int, image_size: int, output_format: str,
                   mix_modes: bool, mode_weights: list,
                   global_start_idx: int = 0, file_seed: int = 42):
    print(f"读取文件: {input_path}")

    vol = tiff.imread(input_path)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(f"期望3D体积 (Z,Y,X)，实际形状: {vol.shape}")

    z, y, x = vol.shape
    dtype = vol.dtype
    print(f"数据形状: {z} slices, {y}x{x} pixels, dtype: {dtype}")

    if max_samples is not None and max_samples < z:
        z = max_samples
        vol = vol[:z]
        print(f"限制为 {z} 切片")

    max_val = _get_max_val(dtype)
    if np.issubdtype(dtype, np.floating):
        p99 = float(np.percentile(vol, 99))
        p1 = float(np.percentile(vol, 1))
        max_val_for_beta = float(p99 - p1 + 1e-8)
    else:
        max_val_for_beta = float(max_val)

    if USE_FIXED_SAVE_RANGE:
        save_lo = float(np.percentile(vol.astype(np.float32), SAVE_RANGE_LOW_PERCENTILE))
        save_hi = float(np.percentile(vol.astype(np.float32), SAVE_RANGE_HIGH_PERCENTILE))
        if save_hi <= save_lo:
            save_lo, save_hi = float(vol.min()), float(vol.max() + 1e-8)
    else:
        save_lo, save_hi = None, None

    rng = np.random.default_rng(file_seed)

    print("生成Z方向参数...")
    alpha_z, beta_z_ratio, env_z = generate_z_profiles(z, rng)
    beta_z_global = beta_z_ratio * max_val_for_beta

    if ENABLE_SPATIAL_FIELDS:
        print("生成空间参数...")
        base_pattern = prepare_spatial_patterns(y, x, rng)
        off_y, off_x = prepare_drift_offsets(z, rng)
    else:
        base_pattern = None
        off_y = off_x = None

    train_idx, val_idx, test_idx = split_indices(z, val_ratio, test_ratio, seed)

    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }

    indices = splits.get(split, train_idx)
    print(f"处理 {split} 集: {len(indices)} 切片")

    clean_dir = os.path.join(output_dir, split, 'clean')
    striped_dir = os.path.join(output_dir, split, 'striped')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(striped_dir, exist_ok=True)

    def select_mode():
        if mix_modes:
            return int(rng.choice(5, p=mode_weights))
        return 0

    print("处理切片并保存...")
    for idx, i in enumerate(tqdm(indices, desc=f"生成{split}数据")):
        img = vol[i].astype(np.float32)

        mode = select_mode()

        if mode == 0:
            sim = apply_feijunyun_mode(
                img, alpha_z, beta_z_global, env_z,
                base_pattern, off_y, off_x, i, rng,
                max_val_for_beta, dtype
            )
        elif mode == 1:
            sim = apply_noise_mode(
                img, alpha_z, beta_z_global, rng,
                max_val_for_beta, dtype
            )
        elif mode == 2:
            sim = apply_basic_mode(
                img, alpha_z, beta_z_global, rng,
                max_val_for_beta, dtype
            )
        elif mode == 3:
            sim = apply_rect_bright_lowcontrast_mode(
                img, rng, max_val_for_beta, dtype
            )
        else:
            sim = apply_full_bright_lowcontrast_mode(
                img, rng, max_val_for_beta, dtype
            )

        global_idx = global_start_idx + idx

        clean_path = os.path.join(clean_dir, f"{split}_{global_idx:06d}.{output_format}")
        save_image(
            vol[i], clean_path,
            image_size if image_size is not None else None,
            save_lo=save_lo, save_hi=save_hi
        )

        striped_path = os.path.join(striped_dir, f"{split}_{global_idx:06d}.{output_format}")
        save_image(
            sim, striped_path,
            image_size if image_size is not None else None,
            save_lo=save_lo, save_hi=save_hi
        )

    print(f"\n{split} 集处理完成!")
    print(f"  干净图像: {clean_dir}")
    print(f"  条纹图像: {striped_dir}")
    print(f"  本次处理样本数: {len(indices)}")

    return len(indices)


def parse_args():
    parser = argparse.ArgumentParser(description='体电镜图像Z轴条纹数据预处理（五种模式混合，已抑制前三种过强黑白饱和，并修复 rect_block 重叠发白）')

    parser.add_argument('--input', '-i', type=str, nargs='+', default=None,
                        help='输入TIF文件路径（支持多个文件或目录）')
    parser.add_argument('--input_dir', '-d', type=str, default=None,
                        help='输入目录（处理目录下所有TIF文件）')
    parser.add_argument('--output_dir', '-o', type=str, default='./data',
                        help='输出数据目录 (默认: ./data)')
    parser.add_argument('--split', type=str, default='all',
                        help='数据划分: train/val/test/all (默认: all)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='每个TIF文件最大样本数 (默认: 全部)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='输出图像尺寸 (默认: 1024)')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'jpg', 'tif'],
                        help='输出图像格式 (默认: png)')
    parser.add_argument('--no_mix', action='store_true',
                        help='不使用混合模式，只使用非均匀场模式')
    parser.add_argument('--weights', type=float, nargs=5,
                        default=[0.30, 0.15, 0.15, 0.15, 0.25],
                        help='五种模式的选择权重，顺序为 feijunyun noise basic rect_block full_slice')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("体电镜图像Z轴条纹数据预处理（五种模式混合）")
    print("已修复：前三种模式过强黑白饱和；rect_block 重叠区域叠加发白")
    print("=" * 60)

    input_files = []

    if args.input:
        for f in args.input:
            if os.path.isfile(f):
                input_files.append(f)
            elif os.path.isdir(f):
                tif_files = [
                    os.path.join(f, x) for x in os.listdir(f)
                    if x.lower().endswith(('.tif', '.tiff'))
                ]
                input_files.extend(tif_files)
            else:
                print(f"警告: 跳过不存在的路径 {f}")

    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
        tif_files = [
            os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir)
            if x.lower().endswith(('.tif', '.tiff'))
        ]
        input_files.extend(tif_files)

    if len(input_files) == 0:
        print("错误: 请提供至少一个输入TIF文件")
        print("用法:")
        print("  python preprocess_em_data_mixed.py --input file1.tif file2.tif --output_dir ./data")
        print("  python preprocess_em_data_mixed.py --input_dir ./tif_folder --output_dir ./data")
        return

    input_files = list(set(input_files))
    input_files.sort()

    print(f"找到 {len(input_files)} 个TIF文件:")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    print()

    for f in input_files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"输入文件不存在: {f}")

    mix_modes = not args.no_mix
    mode_weights = args.weights if mix_modes else [1.0, 0.0, 0.0, 0.0, 0.0]

    mode_weights = np.array(mode_weights, dtype=np.float64)
    sw = mode_weights.sum()
    if sw <= 0:
        raise ValueError("模式权重之和必须大于0")
    mode_weights = (mode_weights / sw).tolist()

    global_start_idx = 0

    for file_idx, input_path in enumerate(input_files):
        print("\n" + "=" * 60)
        print(f"处理第 {file_idx + 1}/{len(input_files)} 个文件: {os.path.basename(input_path)}")
        print("=" * 60)

        file_seed = args.seed + file_idx

        if args.split == 'all':
            for split in ['train', 'val', 'test']:
                count = process_volume(
                    input_path, args.output_dir, split,
                    args.val_ratio, args.test_ratio, args.max_samples,
                    file_seed, args.image_size, args.format,
                    mix_modes, mode_weights,
                    global_start_idx=global_start_idx,
                    file_seed=file_seed
                )
                if split == 'train':
                    global_start_idx += count
        else:
            count = process_volume(
                input_path, args.output_dir, args.split,
                args.val_ratio, args.test_ratio, args.max_samples,
                file_seed, args.image_size, args.format,
                mix_modes, mode_weights,
                global_start_idx=global_start_idx,
                file_seed=file_seed
            )
            global_start_idx += count

    print("\n" + "=" * 60)
    print(f"所有文件处理完成! 共处理 {global_start_idx} 个切片")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()