"""Core paired-data simulator used by vEMDestripe.

This module preserves the implementation invoked by the supplied oblique-
boundary data generator. It includes affine and nonlinear photometric
transformations, local brightness fields, and coherent Z-axis episodes. Each
simulated section remains paired with its unmodified source section.

【核心设计理念】

"无条纹"是相对的 - 一个数据集内图片亮度对比度接近=无条纹，
但放到别的数据集就会产生条纹。

V3 新增：
1. 非线性灰度映射，不再只做 affine
   - gamma 映射
   - 黑白点变化
   - 分段线性 / quantile warp
   - 局部 tone warp
2. coherent large-area field episode
   - 大尺度低频场
   - 分块式场
   - 某个 section 的局部区域持续异常
   - 边界不规则
   - 持续很多张，而不是只做一个小矩形 patch

【工作流程】

数据集A (亮度~80)          数据集B (亮度~180)
    │                            │
    ▼                            ▼
┌─────────┐                 ┌─────────┐
│ 独立模拟 │                 │ 独立模拟 │
│ Z轴条纹  │                 │ Z轴条纹  │
└─────────┘                 └─────────┘
    │                            │
    ▼                            ▼
┌─────────┐                 ┌─────────┐
│ 独立生成 │                 │ 独立生成 │
│ 参考图像 │                 │ 参考图像 │
└─────────┘                 └─────────┘
    │                            │
    └──────────┬─────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ 训练时: 全局归一化 (per-dataset stats)   │
│ 将不同亮度的数据归一化到统一范围           │
└─────────────────────────────────────────┘
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import tifffile as tiff
from PIL import Image
import shutil
import json
from datetime import datetime

# 导入原有条纹模拟函数
from .photometric_ops import (
    generate_z_profiles,
    prepare_spatial_patterns,
    prepare_drift_offsets,
    apply_feijunyun_mode,
    apply_noise_mode,
    apply_basic_mode,
    apply_rect_bright_lowcontrast_mode,
    apply_full_bright_lowcontrast_mode,
    make_rotated_rectangle_mask,
    _get_max_val,
    soft_clip_to_valid_range,
    split_indices,
    AFFINE_STRENGTH_PROBS,
    AFFINE_ALPHA_SMALL, AFFINE_ALPHA_MEDIUM, AFFINE_ALPHA_LARGE,
    AFFINE_BETA_SMALL, AFFINE_BETA_MEDIUM, AFFINE_BETA_LARGE,
    GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX,
    FEIJUNYUN_EXTRA_ALPHA_SCALE, FEIJUNYUN_EXTRA_BETA_SCALE,
    NOISE_EXTRA_ALPHA_SCALE, NOISE_EXTRA_BETA_SCALE,
    BASIC_EXTRA_ALPHA_SCALE, BASIC_EXTRA_BETA_SCALE,
    FINAL_BETA_CLIP_RATIO,
    RECT_MODE_ALPHA_RANGE, RECT_MODE_BETA_RANGE,
    RECT_NUM_RANGE, RECT_W_RATIO_RANGE, RECT_H_RATIO_RANGE,
    RECT_ANGLE_RANGE, RECT_EDGE_BLUR,
    FULL_MODE_ALPHA_RANGE, FULL_MODE_BETA_RANGE, FULL_USE_MEDIAN_CENTER,
    USE_FIXED_SAVE_RANGE, SAVE_RANGE_LOW_PERCENTILE, SAVE_RANGE_HIGH_PERCENTILE,
    USE_LOCAL_BASELINE, BASELINE_SMOOTH,
    MODE_WEIGHTS,
    ENABLE_SPATIAL_FIELDS,
    SOFT_CLIP_LOW_Q, SOFT_CLIP_HIGH_Q, SOFT_CLIP_EXPAND
)

# ============================================================
# 新增参数：非线性灰度映射 + coherent episode + 大矩形异常
# ============================================================

# 新模式数量：
# 原有 5 个 + 新增 5 个（增加大矩形区域异常模式）
NUM_TOTAL_MODES = 10

# 默认模式权重
# 0 feijunyun
# 1 noise
# 2 basic
# 3 rect bright/low contrast (小矩形)
# 4 full bright/low contrast
# 5 gamma
# 6 black/white point
# 7 quantile warp
# 8 local tone warp (coherent large-area field episode)
# 9 rect bright/low contrast (大矩形 512×512 或 700×700)
DEFAULT_MODE_WEIGHTS_V3 = [0.20, 0.09, 0.09, 0.07, 0.13, 0.09, 0.09, 0.07, 0.06, 0.11]

# 每个 volume 额外生成的 episode 数量范围
EPISODE_NUM_RANGE = (2, 6)

# episode 长度占 z 的比例
EPISODE_LEN_RATIO_RANGE = (0.08, 0.35)

# z 边界平滑宽度比例
EPISODE_EDGE_RATIO = 0.25

# gamma 模式
GAMMA_RANGE = (0.70, 1.55)
GAMMA_STRENGTH_RANGE = (0.45, 1.00)

# 黑白点模式
BLACK_SHIFT_RANGE = (0.00, 0.22)
WHITE_SHIFT_RANGE = (0.00, 0.22)
BW_STRENGTH_RANGE = (0.45, 1.00)
FOG_STRENGTH_RANGE = (0.00, 0.18)

# quantile warp 模式
QWARP_MID_SHIFT_RANGE = (-0.18, 0.18)
QWARP_LOW_SHIFT_RANGE = (-0.12, 0.18)
QWARP_HIGH_SHIFT_RANGE = (-0.18, 0.12)
QWARP_STRENGTH_RANGE = (0.45, 1.00)

# local tone warp 模式
LOCAL_TONE_STRENGTH_RANGE = (0.35, 1.00)
LOCAL_GAMMA_RANGE = (0.72, 1.60)
LOCAL_BLACK_SHIFT_RANGE = (0.00, 0.18)
LOCAL_WHITE_SHIFT_RANGE = (0.00, 0.18)
LOCAL_FOG_RANGE = (0.00, 0.25)

# 大尺度 mask 参数
LARGE_MASK_OCCUPANCY_RANGE = (0.22, 0.55)
FIELD_DRIFT_PIX_RANGE = (-24, 24)
MASK_SOFT_BLUR_RATIO = (0.03, 0.10)

# 分块 field 参数
BLOCK_GRID_RANGE = (4, 9)
BLOCK_KEEP_PROB_RANGE = (0.35, 0.75)

# veiling / fog
VEIL_LOW_FREQ_SCALE_RANGE = (0.08, 0.35)

# ============================================================
# 大矩形区域异常参数（新增）
# ============================================================

# 大矩形固定尺寸选项（像素）
LARGE_RECT_SIZES = [512, 700]  # 两种大矩形尺寸
LARGE_RECT_ALPHA_RANGE = (0.65, 0.88)  # 对比度调整范围
LARGE_RECT_BETA_RANGE = (0.06, 0.16)   # 亮度调整范围
LARGE_RECT_EDGE_BLUR = 15  # 边缘模糊（比小矩形更大）

# 保持原始图像的比例（用于训练网络识别正常图像）
KEEP_CLEAN_RATIO = 0.15  # 15% 的切片保持原始状态

# ============================================================
# 基础工具函数
# ============================================================

def sample_mixed_range(rng, probs, small_range, medium_range, large_range):
    """采样混合范围的随机值"""
    probs = np.array(probs, dtype=np.float64)
    probs = probs / probs.sum()
    k = int(rng.choice(3, p=probs))
    if k == 0:
        lo, hi = small_range
    elif k == 1:
        lo, hi = medium_range
    else:
        lo, hi = large_range
    return float(rng.uniform(lo, hi))


def sample_extra_affine(rng, max_val_for_beta):
    """采样额外的仿射变换参数"""
    alpha_delta = sample_mixed_range(
        rng, AFFINE_STRENGTH_PROBS,
        AFFINE_ALPHA_SMALL, AFFINE_ALPHA_MEDIUM, AFFINE_ALPHA_LARGE
    )
    alpha_extra = 1.0 + alpha_delta
    alpha_extra = float(np.clip(alpha_extra, GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX))

    beta_ratio = sample_mixed_range(
        rng, AFFINE_STRENGTH_PROBS,
        AFFINE_BETA_SMALL, AFFINE_BETA_MEDIUM, AFFINE_BETA_LARGE
    )
    beta_extra = float(beta_ratio * max_val_for_beta)
    beta_limit = FINAL_BETA_CLIP_RATIO * max_val_for_beta
    beta_extra = float(np.clip(beta_extra, -beta_limit, beta_limit))

    return alpha_extra, beta_extra


def affine_contrast_brightness(img, alpha, beta, center=None):
    """对比度和亮度仿射变换"""
    img = img.astype(np.float32)
    if center is None:
        center = float(np.median(img))
    return center + alpha * (img - center) + beta


def box_blur_2d_reflect(img, wy, wx):
    """2D盒式模糊"""
    def box_filter_1d_reflect(a, w, axis):
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

    out = box_filter_1d_reflect(img, wy, 0)
    out = box_filter_1d_reflect(out, wx, 1)
    return out.astype(np.float32)


def resize_float_bilinear(img, out_h, out_w):
    """float32 图像双线性 resize"""
    arr = np.asarray(img, dtype=np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        return np.full((out_h, out_w), arr_min, dtype=np.float32)

    norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    pil = Image.fromarray(np.clip(norm * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize((out_w, out_h), Image.BILINEAR)
    out = np.asarray(pil).astype(np.float32) / 255.0
    out = out * (arr_max - arr_min) + arr_min
    return out.astype(np.float32)


def robust_minmax(img, qlo=1.0, qhi=99.0):
    """鲁棒灰度范围"""
    img = img.astype(np.float32)
    lo = float(np.percentile(img, qlo))
    hi = float(np.percentile(img, qhi))
    if hi <= lo:
        lo = float(np.min(img))
        hi = float(np.max(img) + 1e-8)
    return lo, hi


def robust_normalize(img, lo=None, hi=None):
    """按鲁棒范围归一化到 [0,1]"""
    img = img.astype(np.float32)
    if lo is None or hi is None:
        lo, hi = robust_minmax(img)
    out = (img - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0.0, 1.0).astype(np.float32), lo, hi


def robust_denormalize(img01, lo, hi):
    """从 [0,1] 恢复回原始尺度"""
    img01 = np.clip(img01.astype(np.float32), 0.0, 1.0)
    return (img01 * (hi - lo) + lo).astype(np.float32)


def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def ensure_monotonic(points, eps=1e-4):
    """保证控制点单调递增"""
    p = np.asarray(points, dtype=np.float32).copy()
    for i in range(1, len(p)):
        if p[i] < p[i - 1] + eps:
            p[i] = p[i - 1] + eps
    p = np.clip(p, 0.0, 1.0)
    p[-1] = 1.0
    p[0] = 0.0
    for i in range(len(p) - 2, -1, -1):
        if p[i] > p[i + 1] - eps:
            p[i] = max(0.0, p[i + 1] - eps)
    p[0] = 0.0
    p[-1] = 1.0
    return p.astype(np.float32)


def apply_fog_field(img01, fog_strength, fog_field=None):
    """
    veiling / fog:
    将图像往局部亮雾抬升，模拟发灰、发白、低对比
    """
    img01 = img01.astype(np.float32)
    if fog_strength <= 1e-8:
        return img01

    if fog_field is None:
        fog_field = np.ones_like(img01, dtype=np.float32)
    fog_field = np.clip(fog_field.astype(np.float32), 0.0, 1.0)
    veil = fog_strength * fog_field
    out = img01 * (1.0 - 0.45 * veil) + veil
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ============================================================
# 非线性灰度映射
# ============================================================

def gamma_map(img, gamma, fog_strength=0.0, fog_field=None):
    """
    3.1 gamma 映射
    """
    img01, lo, hi = robust_normalize(img)
    out01 = np.power(np.clip(img01, 0.0, 1.0), float(gamma)).astype(np.float32)
    out01 = apply_fog_field(out01, fog_strength, fog_field)
    return robust_denormalize(out01, lo, hi)


def black_white_point_map(img, black_shift, white_shift, fog_strength=0.0, fog_field=None):
    """
    3.2 黑白点变化
    black_shift: 抬黑电平
    white_shift: 压白电平
    """
    img01, lo, hi = robust_normalize(img)
    b = np.clip(float(black_shift), 0.0, 0.45)
    w = np.clip(1.0 - float(white_shift), 0.55, 1.0)
    if w <= b + 1e-4:
        w = min(1.0, b + 0.1)

    out01 = (img01 - b) / (w - b + 1e-8)
    out01 = np.clip(out01, 0.0, 1.0)
    out01 = apply_fog_field(out01, fog_strength, fog_field)
    return robust_denormalize(out01, lo, hi)


def piecewise_quantile_warp_map(img, ctrl_points_dst, fog_strength=0.0, fog_field=None):
    """
    3.3 分段线性 / quantile warp
    输入控制点：
      src 固定为 [0, 0.1, 0.5, 0.9, 1.0]
      dst 由 episode 采样
    """
    img01, lo, hi = robust_normalize(img)
    src = np.array([0.0, 0.10, 0.50, 0.90, 1.0], dtype=np.float32)
    dst = ensure_monotonic(ctrl_points_dst)
    out01 = np.interp(img01, src, dst).astype(np.float32)
    out01 = apply_fog_field(out01, fog_strength, fog_field)
    return robust_denormalize(out01, lo, hi)


def local_tone_warp_map(img, mask, mode_params, fog_field=None):
    """
    3.4 局部 tone warp
    在大尺度 irregular mask 内做局部 tone mapping
    """
    img = img.astype(np.float32)
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

    base = img.copy()
    current = img.copy()

    warp_type = mode_params["warp_type"]
    strength = float(mode_params["strength"])
    fog_strength = float(mode_params.get("fog_strength", 0.0))

    if warp_type == "gamma":
        current = gamma_map(current, mode_params["gamma"], fog_strength=fog_strength, fog_field=fog_field)
    elif warp_type == "bw":
        current = black_white_point_map(
            current,
            mode_params["black_shift"],
            mode_params["white_shift"],
            fog_strength=fog_strength,
            fog_field=fog_field
        )
    elif warp_type == "quantile":
        current = piecewise_quantile_warp_map(
            current,
            mode_params["dst_points"],
            fog_strength=fog_strength,
            fog_field=fog_field
        )
    else:
        # 兜底：局部 affine + fog
        center = float(np.median(current))
        current = affine_contrast_brightness(
            current,
            alpha=mode_params.get("alpha", 0.88),
            beta=mode_params.get("beta", 0.03 * (np.max(current) - np.min(current) + 1e-8)),
            center=center
        )
        cur01, lo, hi = robust_normalize(current)
        cur01 = apply_fog_field(cur01, fog_strength, fog_field)
        current = robust_denormalize(cur01, lo, hi)

    blend = np.clip(mask * strength, 0.0, 1.0)
    out = base * (1.0 - blend) + current * blend
    return out.astype(np.float32)


# ============================================================
# 大尺度低频场 / 分块场 / irregular mask
# ============================================================

def generate_lowfreq_field(h, w, rng, coarse_hw=None, blur_ratio=0.08):
    """
    生成大尺度低频场
    """
    if coarse_hw is None:
        ch = max(4, int(round(h * rng.uniform(0.04, 0.12))))
        cw = max(4, int(round(w * rng.uniform(0.04, 0.12))))
    else:
        ch, cw = coarse_hw

    coarse = rng.normal(0.0, 1.0, size=(ch, cw)).astype(np.float32)
    field = resize_float_bilinear(coarse, h, w)

    by = max(3, int(round(h * blur_ratio)))
    bx = max(3, int(round(w * blur_ratio)))
    if by % 2 == 0:
        by += 1
    if bx % 2 == 0:
        bx += 1

    field = box_blur_2d_reflect(field, by, bx)
    field = field - np.min(field)
    field = field / (np.max(field) - np.min(field) + 1e-8)
    return field.astype(np.float32)


def generate_blocky_field(h, w, rng):
    """
    生成分块式场
    """
    gy = int(rng.integers(BLOCK_GRID_RANGE[0], BLOCK_GRID_RANGE[1] + 1))
    gx = int(rng.integers(BLOCK_GRID_RANGE[0], BLOCK_GRID_RANGE[1] + 1))
    keep_prob = float(rng.uniform(BLOCK_KEEP_PROB_RANGE[0], BLOCK_KEEP_PROB_RANGE[1]))

    coarse = (rng.random((gy, gx)) < keep_prob).astype(np.float32)
    coarse = coarse * rng.uniform(0.4, 1.0, size=(gy, gx)).astype(np.float32)

    field = resize_float_bilinear(coarse, h, w)
    blur_ratio = float(rng.uniform(MASK_SOFT_BLUR_RATIO[0], MASK_SOFT_BLUR_RATIO[1]))
    by = max(3, int(round(h * blur_ratio)))
    bx = max(3, int(round(w * blur_ratio)))
    if by % 2 == 0:
        by += 1
    if bx % 2 == 0:
        bx += 1
    field = box_blur_2d_reflect(field, by, bx)

    field = field - np.min(field)
    field = field / (np.max(field) - np.min(field) + 1e-8)
    return field.astype(np.float32)


def generate_irregular_large_mask(h, w, rng):
    """
    生成 irregular large-area mask：
    - 大尺度低频场
    - 分块式场
    - 可覆盖 1/3 到 1/2 画面
    - 边界不规则，非小矩形
    """
    lowf = generate_lowfreq_field(h, w, rng, blur_ratio=float(rng.uniform(0.06, 0.13)))
    blocky = generate_blocky_field(h, w, rng)
    comb = 0.65 * lowf + 0.35 * blocky

    occupancy = float(rng.uniform(LARGE_MASK_OCCUPANCY_RANGE[0], LARGE_MASK_OCCUPANCY_RANGE[1]))
    thr = float(np.quantile(comb, 1.0 - occupancy))
    mask = (comb >= thr).astype(np.float32)

    blur_ratio = float(rng.uniform(MASK_SOFT_BLUR_RATIO[0], MASK_SOFT_BLUR_RATIO[1]))
    by = max(3, int(round(h * blur_ratio)))
    bx = max(3, int(round(w * blur_ratio)))
    if by % 2 == 0:
        by += 1
    if bx % 2 == 0:
        bx += 1
    mask = box_blur_2d_reflect(mask, by, bx)
    mask = np.clip(mask, 0.0, 1.0)

    # 再乘一个大尺度椭圆-ish 包络，避免太碎
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = float(rng.uniform(0.25, 0.75) * h)
    cx = float(rng.uniform(0.25, 0.75) * w)
    ry = float(rng.uniform(0.28, 0.55) * h)
    rx = float(rng.uniform(0.28, 0.55) * w)
    env = ((yy - cy) / (ry + 1e-8)) ** 2 + ((xx - cx) / (rx + 1e-8)) ** 2
    env = np.exp(-0.5 * env).astype(np.float32)
    env = env / (np.max(env) + 1e-8)

    mask = np.clip(mask * (0.55 + 0.75 * env), 0.0, 1.0)
    return mask.astype(np.float32)


def shift_field(field, dy, dx):
    """用 roll 做轻微漂移"""
    dy = int(round(dy))
    dx = int(round(dx))
    return np.roll(np.roll(field, dy, axis=0), dx, axis=1).astype(np.float32)


# ============================================================
# episode 规划
# ============================================================

def make_z_envelope(z, z0, z1, edge_ratio=EPISODE_EDGE_RATIO):
    """
    生成 z 上平滑包络
    """
    env = np.zeros(z, dtype=np.float32)
    if z1 <= z0:
        return env

    L = max(1, z1 - z0)
    edge = max(1, int(round(L * edge_ratio)))
    for i in range(z):
        if i < z0 or i >= z1:
            env[i] = 0.0
        else:
            if i < z0 + edge:
                t = (i - z0) / max(1.0, edge)
                env[i] = smoothstep01(np.array([t], dtype=np.float32))[0]
            elif i >= z1 - edge:
                t = (z1 - 1 - i) / max(1.0, edge)
                env[i] = smoothstep01(np.array([t], dtype=np.float32))[0]
            else:
                env[i] = 1.0
    return env.astype(np.float32)


def sample_quantile_dst_points(rng):
    """
    为 quantile warp 采样 dst 控制点
    src = [0.0, 0.1, 0.5, 0.9, 1.0]
    """
    low_shift = float(rng.uniform(QWARP_LOW_SHIFT_RANGE[0], QWARP_LOW_SHIFT_RANGE[1]))
    mid_shift = float(rng.uniform(QWARP_MID_SHIFT_RANGE[0], QWARP_MID_SHIFT_RANGE[1]))
    high_shift = float(rng.uniform(QWARP_HIGH_SHIFT_RANGE[0], QWARP_HIGH_SHIFT_RANGE[1]))

    p0 = 0.0
    p1 = np.clip(0.10 + low_shift, 0.01, 0.35)
    p2 = np.clip(0.50 + mid_shift, 0.20, 0.80)
    p3 = np.clip(0.90 + high_shift, 0.65, 0.99)
    p4 = 1.0

    dst = ensure_monotonic([p0, p1, p2, p3, p4])
    return dst.astype(np.float32)


def build_episode_artifacts(z, y, x, rng):
    """
    为整个 volume 规划 coherent artifact episodes
    每个 episode 在 z 上持续一段，在空间上有共享的大尺度 mask / field
    """
    episodes = []

    n_epi = int(rng.integers(EPISODE_NUM_RANGE[0], EPISODE_NUM_RANGE[1] + 1))
    for _ in range(n_epi):
        L = int(round(z * rng.uniform(EPISODE_LEN_RATIO_RANGE[0], EPISODE_LEN_RATIO_RANGE[1])))
        L = int(np.clip(L, 4, max(5, z)))
        z0 = int(rng.integers(0, max(1, z - L + 1)))
        z1 = min(z, z0 + L)
        z_env = make_z_envelope(z, z0, z1)

        mode = int(rng.choice([5, 6, 7, 8], p=[0.26, 0.24, 0.20, 0.30]))

        # 共享 mask / fog field
        base_mask = generate_irregular_large_mask(y, x, rng)
        fog_field = generate_lowfreq_field(y, x, rng, blur_ratio=float(rng.uniform(0.05, 0.14)))

        drift_dy_total = float(rng.uniform(FIELD_DRIFT_PIX_RANGE[0], FIELD_DRIFT_PIX_RANGE[1]))
        drift_dx_total = float(rng.uniform(FIELD_DRIFT_PIX_RANGE[0], FIELD_DRIFT_PIX_RANGE[1]))

        epi = {
            "mode": mode,
            "z0": z0,
            "z1": z1,
            "z_env": z_env,
            "base_mask": base_mask,
            "fog_field": fog_field,
            "drift_dy_total": drift_dy_total,
            "drift_dx_total": drift_dx_total
        }

        if mode == 5:
            epi["gamma"] = float(rng.uniform(GAMMA_RANGE[0], GAMMA_RANGE[1]))
            epi["strength"] = float(rng.uniform(GAMMA_STRENGTH_RANGE[0], GAMMA_STRENGTH_RANGE[1]))
            epi["fog_strength"] = float(rng.uniform(0.0, FOG_STRENGTH_RANGE[1]))
            epi["global_or_local"] = str(rng.choice(["global", "local"], p=[0.55, 0.45]))

        elif mode == 6:
            epi["black_shift"] = float(rng.uniform(BLACK_SHIFT_RANGE[0], BLACK_SHIFT_RANGE[1]))
            epi["white_shift"] = float(rng.uniform(WHITE_SHIFT_RANGE[0], WHITE_SHIFT_RANGE[1]))
            epi["strength"] = float(rng.uniform(BW_STRENGTH_RANGE[0], BW_STRENGTH_RANGE[1]))
            epi["fog_strength"] = float(rng.uniform(0.0, FOG_STRENGTH_RANGE[1]))
            epi["global_or_local"] = str(rng.choice(["global", "local"], p=[0.50, 0.50]))

        elif mode == 7:
            epi["dst_points"] = sample_quantile_dst_points(rng)
            epi["strength"] = float(rng.uniform(QWARP_STRENGTH_RANGE[0], QWARP_STRENGTH_RANGE[1]))
            epi["fog_strength"] = float(rng.uniform(0.0, FOG_STRENGTH_RANGE[1]))
            epi["global_or_local"] = str(rng.choice(["global", "local"], p=[0.60, 0.40]))

        else:
            warp_type = str(rng.choice(["gamma", "bw", "quantile"], p=[0.34, 0.33, 0.33]))
            epi["strength"] = float(rng.uniform(LOCAL_TONE_STRENGTH_RANGE[0], LOCAL_TONE_STRENGTH_RANGE[1]))
            epi["fog_strength"] = float(rng.uniform(LOCAL_FOG_RANGE[0], LOCAL_FOG_RANGE[1]))
            epi["warp_type"] = warp_type
            if warp_type == "gamma":
                epi["gamma"] = float(rng.uniform(LOCAL_GAMMA_RANGE[0], LOCAL_GAMMA_RANGE[1]))
            elif warp_type == "bw":
                epi["black_shift"] = float(rng.uniform(LOCAL_BLACK_SHIFT_RANGE[0], LOCAL_BLACK_SHIFT_RANGE[1]))
                epi["white_shift"] = float(rng.uniform(LOCAL_WHITE_SHIFT_RANGE[0], LOCAL_WHITE_SHIFT_RANGE[1]))
            else:
                epi["dst_points"] = sample_quantile_dst_points(rng)

        episodes.append(epi)

    return episodes


def get_episode_mask_and_fog(epi, i):
    """
    获取某个 slice 的 episode mask / fog_field（含 z 漂移）
    """
    z0 = epi["z0"]
    z1 = epi["z1"]
    if i < z0 or i >= z1:
        return None, None, 0.0

    alpha = float(epi["z_env"][i])
    if alpha <= 1e-8:
        return None, None, 0.0

    rel = 0.0
    if z1 > z0 + 1:
        rel = (i - z0) / float(z1 - z0 - 1)

    dy = epi["drift_dy_total"] * rel
    dx = epi["drift_dx_total"] * rel
    mask = shift_field(epi["base_mask"], dy, dx)
    fog_field = shift_field(epi["fog_field"], 0.6 * dy, 0.6 * dx)

    return mask.astype(np.float32), fog_field.astype(np.float32), alpha


def apply_episode_artifacts(img, i, episodes):
    """
    将 coherent episodes 叠加到当前 slice
    """
    out = img.astype(np.float32)

    for epi in episodes:
        mask, fog_field, z_alpha = get_episode_mask_and_fog(epi, i)
        if z_alpha <= 1e-8:
            continue

        mode = epi["mode"]

        if mode == 5:
            # gamma
            warped = gamma_map(
                out,
                gamma=epi["gamma"],
                fog_strength=epi["fog_strength"] * z_alpha,
                fog_field=fog_field
            )
            if epi["global_or_local"] == "global":
                blend = z_alpha * epi["strength"]
                out = out * (1.0 - blend) + warped * blend
            else:
                blend_mask = np.clip(mask * z_alpha * epi["strength"], 0.0, 1.0)
                out = out * (1.0 - blend_mask) + warped * blend_mask

        elif mode == 6:
            # black/white point
            warped = black_white_point_map(
                out,
                black_shift=epi["black_shift"] * z_alpha,
                white_shift=epi["white_shift"] * z_alpha,
                fog_strength=epi["fog_strength"] * z_alpha,
                fog_field=fog_field
            )
            if epi["global_or_local"] == "global":
                blend = z_alpha * epi["strength"]
                out = out * (1.0 - blend) + warped * blend
            else:
                blend_mask = np.clip(mask * z_alpha * epi["strength"], 0.0, 1.0)
                out = out * (1.0 - blend_mask) + warped * blend_mask

        elif mode == 7:
            # quantile warp
            warped = piecewise_quantile_warp_map(
                out,
                ctrl_points_dst=epi["dst_points"],
                fog_strength=epi["fog_strength"] * z_alpha,
                fog_field=fog_field
            )
            if epi["global_or_local"] == "global":
                blend = z_alpha * epi["strength"]
                out = out * (1.0 - blend) + warped * blend
            else:
                blend_mask = np.clip(mask * z_alpha * epi["strength"], 0.0, 1.0)
                out = out * (1.0 - blend_mask) + warped * blend_mask

        else:
            # local tone warp
            params = {
                "warp_type": epi["warp_type"],
                "strength": epi["strength"] * z_alpha,
                "fog_strength": epi["fog_strength"] * z_alpha
            }
            if epi["warp_type"] == "gamma":
                params["gamma"] = epi["gamma"]
            elif epi["warp_type"] == "bw":
                params["black_shift"] = epi["black_shift"]
                params["white_shift"] = epi["white_shift"]
            else:
                params["dst_points"] = epi["dst_points"]

            out = local_tone_warp_map(out, mask, params, fog_field=fog_field)

    return out.astype(np.float32)


# ============================================================
# 大矩形区域异常模式（新增）
# ============================================================

def make_large_rectangle_mask(h, w, rect_size, cx, cy, blur_w):
    """
    生成大矩形 mask（无旋转，边缘模糊）

    Args:
        h, w: 图像尺寸
        rect_size: 矩形边长（宽高相同）
        cx, cy: 矩形中心坐标
        blur_w: 边缘模糊宽度

    Returns:
        归一化的 mask [0, 1]
    """
    mask = np.zeros((h, w), dtype=np.float32)

    half = rect_size // 2
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half))
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half))

    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1.0

    if blur_w > 0:
        from preprocess_em_data_mixed_morefanwei import box_blur_2d_reflect
        mask = box_blur_2d_reflect(mask, blur_w, blur_w)

    mask = np.clip(mask, 0.0, 1.0)
    return mask


def apply_large_rect_bright_lowcontrast_mode(img: np.ndarray,
                                               rng: np.random.Generator,
                                               max_val_for_beta: float,
                                               dtype):
    """
    大矩形区域异常模式（512×512 或 700×700）

    特点：
    1. 固定尺寸大矩形（512 或 700 像素）
    2. 中心位置随机
    3. 边缘有较大模糊过渡
    4. alpha/beta 范围与小矩形模式相似，但可略大
    """
    img = img.astype(np.float32)
    h, w = img.shape

    from preprocess_em_data_mixed_morefanwei import soft_clip_to_valid_range, clip_by_dtype

    # 随机选择矩形尺寸
    rect_size = int(rng.choice(LARGE_RECT_SIZES))

    # 随机选择中心位置（确保矩形在图像范围内）
    margin = rect_size // 2
    cx = float(rng.uniform(margin, w - margin))
    cy = float(rng.uniform(margin, h - margin))

    # 生成 mask
    mask = make_large_rectangle_mask(h, w, rect_size, cx, cy, LARGE_RECT_EDGE_BLUR)

    # 采样变换参数
    alpha = float(rng.uniform(LARGE_RECT_ALPHA_RANGE[0], LARGE_RECT_ALPHA_RANGE[1]))
    beta = float(rng.uniform(LARGE_RECT_BETA_RANGE[0], LARGE_RECT_BETA_RANGE[1])) * max_val_for_beta

    center = float(np.median(img))
    transformed = affine_contrast_brightness(img, alpha=alpha, beta=beta, center=center)

    sim = (1.0 - mask) * img + mask * transformed
    sim = soft_clip_to_valid_range(sim, img)
    sim = clip_by_dtype(sim, dtype, max_val_for_beta)
    return sim


# ============================================================
# 参考图像选择与融合
# ============================================================

def select_random_images_for_reference(volume, num_select=50, seed=42):
    """
    随机选择图像作为参考图像来源

    【核心逻辑】
    从数据集中随机抽取指定数量的图像用于生成参考图。
    这些图像的融合结果将反映数据集的整体亮度/对比度分布，
    但随后会进行结构破坏以避免网络学习到结构信息。
    """
    rng = np.random.default_rng(seed)

    n = len(volume)
    # 随机选择num_select张图像
    selected_indices = rng.choice(n, size=min(num_select, n), replace=False)
    selected_indices = sorted(selected_indices.tolist())

    # 打印选择的统计信息供参考
    brightnesses = [np.mean(volume[i]) for i in selected_indices[:10]]
    print(f"  随机选择 {len(selected_indices)} 张图像（亮度范围: {min(brightnesses):.1f} - {max(brightnesses):.1f}）")

    return selected_indices


def select_normal_images_by_stats(volume, num_select=50, seed=42):
    """
    根据统计特征选择"正常"图像

    【核心逻辑】
    在同一个数据集中，选择亮度/对比度接近中值的图像作为"正常"图像。
    这些图像的亮度/对比度与数据集的整体分布一致。

    注意：此函数已不推荐使用，请使用 select_random_images_for_reference
    """
    rng = np.random.default_rng(seed)

    n = len(volume)
    brightnesses = np.zeros(n, dtype=np.float32)
    contrasts = np.zeros(n, dtype=np.float32)

    for i in range(n):
        img = volume[i].astype(np.float32)
        brightnesses[i] = np.mean(img)
        contrasts[i] = np.std(img)

    median_brightness = np.median(brightnesses)
    median_contrast = np.median(contrasts)

    distances = np.sqrt(
        (brightnesses - median_brightness) ** 2 +
        (contrasts - median_contrast) ** 2
    )

    selected_indices = np.argsort(distances)[:num_select]
    selected_indices = sorted(selected_indices.tolist())

    print(f"  正常图像选择: 中值亮度={median_brightness:.1f}, 中值对比度={median_contrast:.1f}")
    print(f"  选中 {len(selected_indices)} 张正常图像")

    return selected_indices


def destroy_structure(image, block_size=32, num_noisy_blocks=0):
    """
    破坏图像的结构信息，只保留整体亮度分布

    【方法】
    1. 将图像分成block_size x block_size的块
    2. 计算每个块的均值
    3. 用块均值替换原图像，完全去除纹理和结构
    4. 可选：添加少量噪声增加多样性

    【目的】
    确保网络不会通过Reference学习到细胞结构信息，
    Reference只提供亮度/对比度的统计参考。

    Args:
        image: 输入图像 (H, W), float32, 0-255范围
        block_size: 块大小，默认32像素
        num_noisy_blocks: 保留多少个原始块(带结构)以增加多样性

    Returns:
        破坏结构后的图像
    """
    h, w = image.shape
    result = np.zeros_like(image)

    # 计算块边界
    h_blocks = (h + block_size - 1) // block_size
    w_blocks = (w + block_size - 1) // block_size

    # 创建块的随机顺序，用于选择保留原始结构的块
    rng = np.random.default_rng(42)
    block_indices = list(range(h_blocks * w_blocks))
    rng.shuffle(block_indices)
    keep_struct_blocks = set(block_indices[:num_noisy_blocks])

    block_id = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            # 块的边界
            h_start = i * block_size
            h_end = min((i + 1) * block_size, h)
            w_start = j * block_size
            w_end = min((j + 1) * block_size, w)

            block = image[h_start:h_end, w_start:w_end]

            if block_id in keep_struct_blocks:
                # 保留少量原始块，增加多样性
                result[h_start:h_end, w_start:w_end] = block
            else:
                # 用块的均值替换，完全破坏结构
                block_mean = np.mean(block)
                result[h_start:h_end, w_start:w_end] = block_mean

            block_id += 1

    return result


def create_reference_image(images_list, target_size=None, destroy_struct=True, block_size=32):
    """
    从多张随机选择的图像创建融合参考图像，并破坏结构信息

    【核心改进】
    1. 使用随机选择的图像（而非"最正常"的图像）
    2. 融合后进行结构破坏，只保留亮度分布

    【目的】
    Reference图像只提供亮度/对比度的统计参考，
    不包含任何可用于学习细胞结构的细节信息。
    """
    if len(images_list) == 0:
        return None

    if target_size is None:
        target_size = images_list[0].shape[:2]

    # 1. 融合多张图像
    fused = np.zeros((target_size[0], target_size[1]), dtype=np.float32)

    for img in images_list:
        if img.shape[:2] != target_size:
            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            img = np.array(pil_img)

        fused += img.astype(np.float32)

    fused = fused / len(images_list)

    # 2. 破坏结构信息
    if destroy_struct:
        fused = destroy_structure(fused, block_size=block_size)

    return fused.astype(np.uint8)


# ============================================================
# 主处理逻辑
# ============================================================

def process_single_volume_for_training(
    input_path,
    output_dir,
    split,
    val_ratio,
    test_ratio,
    max_samples,
    seed,
    image_size,
    output_format,
    mix_modes,
    mode_weights,
    reference_num_images,
    file_seed,
    global_start_idx=0,
    dataset_name=None,
    keep_clean_ratio=KEEP_CLEAN_RATIO
):
    """
    处理单个 TIF 文件，生成训练数据

    输出目录结构保持不变:
    output_dir/
    ├── dataset_A/
    │   ├── train/
    │   │   ├── clean/
    │   │   ├── striped/
    │   │   └── reference/
    │   ├── val/
    │   └── test/
    """
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(input_path))[0]

    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")

    vol = tiff.imread(input_path)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(f"期望3D体积 (Z,Y,X)，实际形状: {vol.shape}")

    z, y, x = vol.shape
    dtype = vol.dtype
    print(f"  数据形状: {z} slices, {y}x{x} pixels, dtype: {dtype}")

    vol_flat = vol.astype(np.float32).reshape(z, -1)
    dataset_mean = np.mean(vol_flat)
    dataset_std = np.std(vol_flat)
    print(f"  数据集统计: 均值={dataset_mean:.2f}, 标准差={dataset_std:.2f}")

    if max_samples is not None and max_samples < z:
        z = max_samples
        vol = vol[:z]
        print(f"  限制为 {z} 切片")

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

    print("  生成原有Z方向参数...")
    alpha_z, beta_z_ratio, env_z = generate_z_profiles(z, rng)
    beta_z_global = beta_z_ratio * max_val_for_beta

    if ENABLE_SPATIAL_FIELDS:
        print("  生成原有空间参数...")
        base_pattern = prepare_spatial_patterns(y, x, rng)
        off_y, off_x = prepare_drift_offsets(z, rng)
    else:
        base_pattern = None
        off_y = off_x = None

    # 新增 coherent episodes
    print("  生成新增 coherent artifact episodes...")
    artifact_episodes = build_episode_artifacts(z, y, x, rng)
    print(f"  新增 episode 数量: {len(artifact_episodes)}")

    train_idx, val_idx, test_idx = split_indices(z, val_ratio, test_ratio, seed)

    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }

    indices = splits.get(split, train_idx)
    print(f"  处理 {split} 集: {len(indices)} 切片")

    clean_dir = os.path.join(output_dir, dataset_name, split, 'clean')
    striped_dir = os.path.join(output_dir, dataset_name, split, 'striped')
    reference_dir = os.path.join(output_dir, dataset_name, split, 'reference')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(striped_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)

    def select_mode():
        if mix_modes:
            return int(rng.choice(NUM_TOTAL_MODES, p=mode_weights))
        return 0

    def save_image(img, path, image_size=None):
        """
        保存图像为指定格式
        存储方式与原来保持一致
        """
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

    def apply_original_or_new_mode(img, i, rng):
        """
        原有模式 + 新增非线性模式
        再叠加 coherent episode
        """
        mode = select_mode()
        img = img.astype(np.float32)

        # ---------- 第一层：基础模式 ----------
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

        elif mode == 4:
            sim = apply_full_bright_lowcontrast_mode(
                img, rng, max_val_for_beta, dtype
            )

        elif mode == 5:
            # 新增 gamma 映射
            gamma = float(rng.uniform(GAMMA_RANGE[0], GAMMA_RANGE[1]))
            fog_field = generate_lowfreq_field(y, x, rng, blur_ratio=float(rng.uniform(0.05, 0.12)))
            fog_strength = float(rng.uniform(0.0, FOG_STRENGTH_RANGE[1]))
            sim = gamma_map(img, gamma=gamma, fog_strength=fog_strength, fog_field=fog_field)

        elif mode == 6:
            # 新增 黑白点变化
            fog_field = generate_lowfreq_field(y, x, rng, blur_ratio=float(rng.uniform(0.05, 0.12)))
            sim = black_white_point_map(
                img,
                black_shift=float(rng.uniform(BLACK_SHIFT_RANGE[0], BLACK_SHIFT_RANGE[1])),
                white_shift=float(rng.uniform(WHITE_SHIFT_RANGE[0], WHITE_SHIFT_RANGE[1])),
                fog_strength=float(rng.uniform(0.0, FOG_STRENGTH_RANGE[1])),
                fog_field=fog_field
            )

        elif mode == 7:
            # 新增 分段线性 / quantile warp
            fog_field = generate_lowfreq_field(y, x, rng, blur_ratio=float(rng.uniform(0.05, 0.12)))
            sim = piecewise_quantile_warp_map(
                img,
                ctrl_points_dst=sample_quantile_dst_points(rng),
                fog_strength=float(rng.uniform(0.0, FOG_STRENGTH_RANGE[1])),
                fog_field=fog_field
            )

        elif mode == 8:
            # 新增 局部 tone warp：单张 fallback 版本
            mask = generate_irregular_large_mask(y, x, rng)
            fog_field = generate_lowfreq_field(y, x, rng, blur_ratio=float(rng.uniform(0.05, 0.12)))
            warp_type = str(rng.choice(["gamma", "bw", "quantile"], p=[0.34, 0.33, 0.33]))
            params = {
                "warp_type": warp_type,
                "strength": float(rng.uniform(LOCAL_TONE_STRENGTH_RANGE[0], LOCAL_TONE_STRENGTH_RANGE[1])),
                "fog_strength": float(rng.uniform(LOCAL_FOG_RANGE[0], LOCAL_FOG_RANGE[1]))
            }
            if warp_type == "gamma":
                params["gamma"] = float(rng.uniform(LOCAL_GAMMA_RANGE[0], LOCAL_GAMMA_RANGE[1]))
            elif warp_type == "bw":
                params["black_shift"] = float(rng.uniform(LOCAL_BLACK_SHIFT_RANGE[0], LOCAL_BLACK_SHIFT_RANGE[1]))
                params["white_shift"] = float(rng.uniform(LOCAL_WHITE_SHIFT_RANGE[0], LOCAL_WHITE_SHIFT_RANGE[1]))
            else:
                params["dst_points"] = sample_quantile_dst_points(rng)

            sim = local_tone_warp_map(img, mask, params, fog_field=fog_field)

        elif mode == 9:
            # 新增 大矩形区域异常（512×512 或 700×700）
            sim = apply_large_rect_bright_lowcontrast_mode(
                img, rng, max_val_for_beta, dtype
            )

        # ---------- 第二层：叠加 coherent episode ----------
        sim = apply_episode_artifacts(sim, i, artifact_episodes)

        # ---------- 最终裁剪 ----------
        sim = soft_clip_to_valid_range(sim, img)
        return sim.astype(np.float32)

    print(f"  随机选择图像作为参考...")
    all_random_indices = select_random_images_for_reference(vol, num_select=reference_num_images, seed=file_seed)

    normal_train = [i for i in all_random_indices if i in train_idx][:reference_num_images // 3 or 10]
    normal_val = [i for i in all_random_indices if i in val_idx][:reference_num_images // 3 or 10]
    normal_test = [i for i in all_random_indices if i in test_idx][:reference_num_images // 3 or 10]

    split_normals = {
        'train': normal_train,
        'val': normal_val,
        'test': normal_test
    }

    reference_images = []
    for idx_ref in split_normals[split]:
        ref_img = vol[idx_ref].astype(np.uint8)
        if image_size is not None:
            pil_img = Image.fromarray(ref_img)
            pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
            ref_img = np.array(pil_img)
        reference_images.append(ref_img)

    print(f"  参考图像将进行结构破坏处理...")

    print(f"  处理切片并保存...")

    # 随机选择保持原始状态的切片索引
    num_indices = len(indices)
    num_keep_clean = int(num_indices * keep_clean_ratio)
    keep_clean_set = set(rng.choice(indices, size=min(num_keep_clean, num_indices), replace=False).tolist())
    print(f"  保持原始状态: {len(keep_clean_set)} 张 ({keep_clean_ratio*100:.1f}%)")

    for idx_local, i in enumerate(tqdm(indices, desc=f"生成{split}数据")):
        img = vol[i].astype(np.float32)

        # 如果该切片被选中保持原始状态，直接使用原图
        if i in keep_clean_set:
            sim = img.copy()
        else:
            sim = apply_original_or_new_mode(img, i, rng)

        global_idx = global_start_idx + idx_local

        clean_path = os.path.join(clean_dir, f"{split}_{global_idx:06d}.{output_format}")
        save_image(vol[i], clean_path, image_size if image_size else None)

        striped_path = os.path.join(striped_dir, f"{split}_{global_idx:06d}.{output_format}")
        save_image(sim, striped_path, image_size if image_size else None)

    if len(reference_images) > 0:
        target_size = (image_size, image_size) if image_size else reference_images[0].shape[:2]
        # 融合图像并破坏结构，只保留亮度分布信息
        fused_ref = create_reference_image(reference_images, target_size, destroy_struct=True, block_size=32)

        if fused_ref is not None:
            ref_path = os.path.join(reference_dir, f"{split}_reference_fused.{output_format}")
            pil_ref = Image.fromarray(fused_ref)
            pil_ref.save(ref_path)

            for idx_ref, ref_img in enumerate(reference_images):
                ref_single_path = os.path.join(reference_dir, f"{split}_ref_{idx_ref:03d}.{output_format}")
                pil_single = Image.fromarray(ref_img)
                pil_single.save(ref_single_path)

            print(f"  生成了 {len(reference_images)} 张参考图像")
            print(f"  融合后进行结构破坏 (block_size=32)")
            print(f"  保存到: {ref_path}")

    print(f"\n  {split} 集处理完成!")
    print(f"    Clean: {clean_dir}")
    print(f"    Striped: {striped_dir}")
    print(f"    Reference: {reference_dir}")

    return len(indices)


# ============================================================
# 数据集信息
# ============================================================

def generate_dataset_info(output_dir, datasets, splits=['train', 'val', 'test']):
    """生成数据集信息 JSON"""
    info = {
        'created_at': datetime.now().isoformat(),
        'description': '每个数据集独立处理，参考图像来自同一数据集内部；V3 增加非线性灰度映射与 coherent large-area field episodes',
        'datasets': {},
        'splits': {}
    }

    for dataset_name in datasets:
        dataset_info = {
            'name': dataset_name,
            'splits': {}
        }

        for split in splits:
            clean_dir = os.path.join(output_dir, dataset_name, split, 'clean')
            striped_dir = os.path.join(output_dir, dataset_name, split, 'striped')
            reference_dir = os.path.join(output_dir, dataset_name, split, 'reference')

            clean_files = sorted(glob(os.path.join(clean_dir, '*g'))) if os.path.exists(clean_dir) else []
            striped_files = sorted(glob(os.path.join(striped_dir, '*g'))) if os.path.exists(striped_dir) else []
            reference_files = sorted(glob(os.path.join(reference_dir, '*g'))) if os.path.exists(reference_dir) else []

            dataset_info['splits'][split] = {
                'num_samples': len(clean_files),
                'num_reference_images': len([f for f in reference_files if 'reference_fused' not in f])
            }

            info['splits'][f"{dataset_name}/{split}"] = {
                'num_samples': len(clean_files),
                'clean_dir': clean_dir,
                'striped_dir': striped_dir,
                'reference_dir': reference_dir,
                'reference_fused': os.path.join(reference_dir, f"{split}_reference_fused.png") if reference_files else None,
                'note': '参考图像来自同一数据集；新增非线性条纹模拟'
            }

        info['datasets'][dataset_name] = dataset_info

    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n数据集信息已保存到: {info_path}")
    return info


# ============================================================
# 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='体电镜Z轴条纹去除 - 训练数据生成 V3\n\n'
                    '【核心设计】\n'
                    '每个数据集独立处理：\n'
                    '1. 条纹模拟在数据集内部独立进行\n'
                    '2. 参考图像从同一数据集的"正常"图像中选择\n'
                    '3. 训练时会将不同数据集归一化到统一范围\n'
                    '4. 新增非线性灰度映射与 coherent large-area field episode\n\n'
                    '【关键概念】\n'
                    '"无条纹"是相对的 - 数据集内图片亮度一致≠跨数据集亮度一致\n'
                    '每个数据集有自己的亮度/对比度基准',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理目录中的所有TIF文件（每个文件作为独立数据集）
  python prepare_training_data_v3.py --input_dir ./clean_tifs --output_dir ./training_data

  # 处理多个特定TIF文件
  python prepare_training_data_v3.py --input data1.tif data2.tif --output_dir ./training_data

  # 自定义参数
  python prepare_training_data_v3.py --input_dir ./clean_tifs \\
      --output_dir ./training_data \\
      --image_size 512 \\
      --reference_num 50

  # 使用 9 个模式权重（推荐）
  python prepare_training_data_v3.py --input_dir ./clean_tifs \\
      --output_dir ./training_data \\
      --weights 0.22 0.10 0.10 0.08 0.15 0.10 0.10 0.08 0.07

  # 兼容旧版 5 个模式权重
  python prepare_training_data_v3.py --input_dir ./clean_tifs \\
      --output_dir ./training_data \\
      --weights 0.30 0.15 0.15 0.15 0.25
        """
    )

    parser.add_argument('--input', '-i', type=str, nargs='+', default=None,
                        help='输入TIF文件路径（支持多个文件）')
    parser.add_argument('--input_dir', '-d', type=str, default=None,
                        help='输入目录（处理目录下所有TIF文件，每个TIF作为独立数据集）')
    parser.add_argument('--output_dir', '-o', type=str, default='./training_data',
                        help='输出数据目录 (默认: ./training_data)')
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
                        help='不使用混合模式，只使用 mode 0 的基础非均匀场模式，再叠加 coherent episodes')
    parser.add_argument('--weights', type=float, nargs='+',
                        default=DEFAULT_MODE_WEIGHTS_V3,
                        help='模式权重；可传 5 个（旧版）或 10 个（V3 推荐，含大矩形异常模式）')
    parser.add_argument('--reference_num', type=int, default=50,
                        help='每个split生成的参考图像数量 (默认: 50)')

    parser.add_argument('--keep_clean_ratio', type=float, default=KEEP_CLEAN_RATIO,
                        help=f'保持原始图像的比例 (默认: {KEEP_CLEAN_RATIO})')

    # 【新增】局部模拟控制参数
    parser.add_argument('--disable_local_tone_warp', action='store_true',
                        help='禁用局部tone warp模式（模式8）')
    parser.add_argument('--enable_local_tone_warp', action='store_true',
                        help='启用局部tone warp模式（模式8）')

    return parser.parse_args()


# ============================================================
# 主函数
# ============================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("体电镜Z轴条纹去除 - 训练数据生成 V3")
    print("=" * 60)
    print()
    print("【核心设计】")
    print("每个数据集独立处理:")
    print("  1. 条纹模拟在数据集内部独立进行")
    print("  2. 参考图像从同一数据集的'正常'图像中选择")
    print("  3. 不同亮度的数据集会被分别保存")
    print("  4. 新增非线性灰度映射和 coherent large-area field episodes")
    print()

    input_files = []

    if args.input:
        for f in args.input:
            if os.path.isfile(f):
                input_files.append(f)
            else:
                print(f"警告: 跳过不存在的文件 {f}")

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
        print("  python prepare_training_data_v3.py --input_dir ./clean_tifs --output_dir ./data")
        return

    input_files = list(set(input_files))
    input_files.sort()

    print(f"找到 {len(input_files)} 个TIF文件 (将作为 {len(input_files)} 个独立数据集处理):")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    print()

    mix_modes = not args.no_mix

    # 兼容旧版 5 权重，自动补齐成 9 权重
    raw_weights = list(args.weights)
    if len(raw_weights) == 5:
        # 前 5 个沿用，后 4 个给一个温和默认值
        raw_weights = raw_weights + [0.12, 0.12, 0.10, 0.11]
    elif len(raw_weights) != NUM_TOTAL_MODES:
        raise ValueError(f"--weights 需要传 5 个（旧版兼容）或 {NUM_TOTAL_MODES} 个（V3），当前为 {len(raw_weights)} 个")

    # 处理局部模拟参数
    if args.disable_local_tone_warp:
        # 禁用模式8 (local tone warp)
        raw_weights[8] = 0.0

    if args.enable_local_tone_warp:
        # 提高模式8的权重
        raw_weights[8] = max(raw_weights[8], 0.30)

    mode_weights = raw_weights if mix_modes else [1.0] + [0.0] * (NUM_TOTAL_MODES - 1)
    mode_weights = np.array(mode_weights, dtype=np.float64)
    sw = mode_weights.sum()
    if sw <= 0:
        raise ValueError("模式权重之和必须大于0")
    mode_weights = (mode_weights / sw).tolist()

    print("模式权重:")
    for k, v in enumerate(mode_weights):
        print(f"  mode {k}: {v:.4f}")
    print()

    global_start_idx = 0
    processed_datasets = []

    for file_idx, input_path in enumerate(input_files):
        dataset_name = os.path.splitext(os.path.basename(input_path))[0]
        processed_datasets.append(dataset_name)

        file_seed = args.seed + file_idx

        if args.split == 'all':
            for split in ['train', 'val', 'test']:
                count = process_single_volume_for_training(
                    input_path, args.output_dir, split,
                    args.val_ratio, args.test_ratio, args.max_samples,
                    file_seed, args.image_size, args.format,
                    mix_modes, mode_weights,
                    reference_num_images=args.reference_num,
                    file_seed=file_seed,
                    global_start_idx=global_start_idx,
                    dataset_name=dataset_name
                )
                if split == 'train':
                    global_start_idx += count
        else:
            count = process_single_volume_for_training(
                input_path, args.output_dir, args.split,
                args.val_ratio, args.test_ratio, args.max_samples,
                file_seed, args.image_size, args.format,
                mix_modes, mode_weights,
                reference_num_images=args.reference_num,
                file_seed=file_seed,
                global_start_idx=global_start_idx,
                dataset_name=dataset_name
            )
            global_start_idx += count

    if args.split == 'all':
        generate_dataset_info(args.output_dir, processed_datasets)

    print("\n" + "=" * 60)
    print(f"所有文件处理完成! 共处理 {global_start_idx} 个切片，来自 {len(processed_datasets)} 个数据集")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    print("\n生成的数据集结构:")
    print(f"""每个数据集独立目录:
{args.output_dir}/
├── dataset_A/
│   ├── train/
│   │   ├── clean/
│   │   ├── striped/
│   │   └── reference/
│   ├── val/
│   └── test/
├── dataset_B/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset_C/
│   └── ...
└── dataset_info.json

【关键说明】
- 每个数据集有独立的参考图像（来自同一数据集）
- 训练时，不同数据集会被归一化到统一范围
- 推理时，参考图像必须来自待处理的数据集
- V3 新增：
  * gamma 映射
  * 黑白点变化
  * 分段线性 / quantile warp
  * 局部 tone warp
  * coherent large-area field episodes
""")


if __name__ == "__main__":
    main()
