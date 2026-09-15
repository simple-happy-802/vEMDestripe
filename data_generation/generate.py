"""Generate the paired synthetic training data used by vEMDestripe.

This entry point reuses the stable loading, splitting, reference generation,
coherent-episode, and saving code from ``prepare_training_data_v_3_nonlinear``.
It changes only the parts requested by the Bumbarger real-volume analysis:

1. Use the ten-mode weights
   [0.22, 0.09, 0.11, 0.07, 0.13, 0.09, 0.09, 0.14, 0.06, 0.00].
2. Replace M3 small rectangles with a large oblique half-plane bright veil.
3. Forward ``--keep_clean_ratio`` correctly to the underlying generator.

The new M3 is based on the abnormal/adjacent pairs 197/198 and 241/242:
the bright region covers roughly 80% of the frame, its straight boundary is
oriented at about 36 or 68 degrees, and the affected texture retains only
about 20-45% of its original contrast.  The implementation samples around
those observations instead of copying the test images themselves.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import core as base
from .photometric_ops import clip_by_dtype


# M0   M1   M2   M3   M4   M5   M6   M7   M8   M9
DEFAULT_MODE_WEIGHTS_OBLIQUE = [
    0.22, 0.09, 0.11, 0.07, 0.13,
    0.09, 0.09, 0.14, 0.06, 0.00,
]

# Data-driven M3 parameter ranges.  Angles describe the visible boundary line,
# not its normal.  Both supplied examples have a positive-slope oblique edge.
OBLIQUE_BOUNDARY_ANGLE_RANGE_DEG = (30.0, 72.0)
OBLIQUE_OCCUPANCY_RANGE = (0.72, 0.88)
OBLIQUE_EDGE_WIDTH_RATIO_RANGE = (0.006, 0.020)
OBLIQUE_CONTRAST_REMAIN_RANGE = (0.18, 0.45)
OBLIQUE_VEIL_LEVEL_RANGE = (0.72, 0.92)
OBLIQUE_FIELD_AMPLITUDE_RANGE = (0.00, 0.025)


def make_oblique_halfplane_mask(
    h: int,
    w: int,
    rng: np.random.Generator,
):
    """Create a large soft-edged oblique half-plane mask.

    The threshold is selected by a score quantile, so ``occupancy`` controls
    the affected fraction directly even for rectangular images.  A random
    sign chooses which side of the line is brightened while preserving the
    observed boundary-angle family.
    """
    boundary_angle_deg = float(rng.uniform(*OBLIQUE_BOUNDARY_ANGLE_RANGE_DEG))
    occupancy = float(rng.uniform(*OBLIQUE_OCCUPANCY_RANGE))
    edge_width_px = float(
        rng.uniform(*OBLIQUE_EDGE_WIDTH_RATIO_RANGE) * min(h, w)
    )

    theta = np.deg2rad(boundary_angle_deg)
    # Unit normal to a line whose direction is theta.
    normal_x = -np.sin(theta)
    normal_y = np.cos(theta)
    if rng.random() < 0.5:
        normal_x = -normal_x
        normal_y = -normal_y

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx -= 0.5 * (w - 1)
    yy -= 0.5 * (h - 1)
    signed_distance = normal_x * xx + normal_y * yy

    # score >= threshold occupies approximately the requested frame fraction.
    threshold = float(np.quantile(signed_distance, 1.0 - occupancy))
    edge_width_px = max(1.0, edge_width_px)
    t = 0.5 + 0.5 * (signed_distance - threshold) / edge_width_px
    t = np.clip(t, 0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)

    metadata = {
        "boundary_angle_deg": boundary_angle_deg,
        "occupancy": occupancy,
        "edge_width_px": edge_width_px,
        "bright_side_sign": 1 if normal_y >= 0.0 else -1,
    }
    return mask.astype(np.float32), metadata


def apply_oblique_halfplane_veil_mode(
    img: np.ndarray,
    rng: np.random.Generator,
    max_val_for_beta: float,
    dtype,
    return_metadata: bool = False,
):
    """Apply the replacement M3: oblique bright veil with contrast loss.

    Inside the half-plane, the image is blended toward a bright veil level:

        affected = contrast_remain * image
                   + (1 - contrast_remain) * veil

    This retains ultrastructure but compresses its contrast, matching the two
    supplied abnormal slices more closely than a small rectangular offset.
    """
    src = np.asarray(img, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"M3 expects a 2-D grayscale image, got {src.shape}")

    h, w = src.shape
    mask, metadata = make_oblique_halfplane_mask(h, w, rng)
    contrast_remain = float(rng.uniform(*OBLIQUE_CONTRAST_REMAIN_RANGE))
    veil_level = float(rng.uniform(*OBLIQUE_VEIL_LEVEL_RANGE))
    field_amplitude = float(rng.uniform(*OBLIQUE_FIELD_AMPLITUDE_RANGE))

    # Work in the slice's robust 1st-99th percentile range.  The base saver
    # later maps the volume-wide 1st-99th percentile range to uint8.  Using
    # the dtype maximum here would therefore be stretched a second time and
    # turn a realistic veil into a clipped white polygon.
    src01, robust_lo, robust_hi = base.robust_normalize(src)

    if field_amplitude > 1e-8:
        low_frequency = base.generate_lowfreq_field(
            h,
            w,
            rng,
            blur_ratio=float(rng.uniform(0.08, 0.16)),
        )
        veil_map = veil_level + field_amplitude * (2.0 * low_frequency - 1.0)
    else:
        veil_map = np.full_like(src01, veil_level, dtype=np.float32)
    veil_map = np.clip(veil_map, 0.0, 1.0)

    affected01 = (
        contrast_remain * src01
        + (1.0 - contrast_remain) * veil_map
    )
    out01 = src01 * (1.0 - mask) + affected01 * mask
    sim = base.robust_denormalize(out01, robust_lo, robust_hi)

    # The base pipeline applies soft_clip_to_valid_range after the optional
    # coherent episode.  Do not apply it here as well: double soft-clipping
    # made the simulated bright field substantially darker than slices 197
    # and 241 during calibration.
    sim = clip_by_dtype(sim, dtype, max_val_for_beta)

    metadata.update(
        {
            "contrast_remain": contrast_remain,
            "veil_level": veil_level,
            "field_amplitude": field_amplitude,
        }
    )
    if return_metadata:
        return sim.astype(np.float32), metadata
    return sim.astype(np.float32)


# Patch the imported implementation before invoking its normal CLI pipeline.
# The nested per-slice dispatch in the base generator resolves these globals at
# runtime, so all other modes and I/O behavior stay unchanged.
base.DEFAULT_MODE_WEIGHTS_V3 = list(DEFAULT_MODE_WEIGHTS_OBLIQUE)
base.apply_rect_bright_lowcontrast_mode = apply_oblique_halfplane_veil_mode


# The original CLI defines --keep_clean_ratio but does not pass it into
# process_single_volume_for_training.  Capture and forward it in this entry
# point so the documented option actually works.
_original_process_single_volume = base.process_single_volume_for_training
_active_keep_clean_ratio = base.KEEP_CLEAN_RATIO


def _parse_args_and_capture_clean_ratio():
    global _active_keep_clean_ratio
    parser = argparse.ArgumentParser(
        description=(
            "V3 nonlinear training-data generation with a data-driven M3 "
            "oblique bright-veil boundary"
        )
    )
    parser.add_argument("--input", "-i", type=str, nargs="+", default=None)
    parser.add_argument("--input_dir", "-d", type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default="./training_data")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument(
        "--format",
        type=str,
        choices=("png", "jpg", "tif"),
        default="png",
    )
    parser.add_argument("--no_mix", action="store_true")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=base.NUM_TOTAL_MODES,
        default=list(DEFAULT_MODE_WEIGHTS_OBLIQUE),
        metavar=("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"),
        help="exactly ten non-negative weights; they are normalized internally",
    )
    parser.add_argument("--reference_num", type=int, default=50)
    parser.add_argument(
        "--keep_clean_ratio",
        type=float,
        default=base.KEEP_CLEAN_RATIO,
    )
    parser.add_argument("--disable_local_tone_warp", action="store_true")
    parser.add_argument("--enable_local_tone_warp", action="store_true")
    args = parser.parse_args()

    ratio = float(args.keep_clean_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("--keep_clean_ratio must be between 0 and 1")
    if any(float(weight) < 0.0 for weight in args.weights):
        raise ValueError("--weights cannot contain negative values")
    _active_keep_clean_ratio = ratio
    return args


def _process_single_volume_with_clean_ratio(*args, **kwargs):
    kwargs.setdefault("keep_clean_ratio", _active_keep_clean_ratio)
    return _original_process_single_volume(*args, **kwargs)


base.parse_args = _parse_args_and_capture_clean_ratio
base.process_single_volume_for_training = _process_single_volume_with_clean_ratio


def main():
    print("vEMDestripe paired-data simulation")
    print("M3: oblique half-plane bright veil / low contrast")
    print(f"Default mode weights: {DEFAULT_MODE_WEIGHTS_OBLIQUE}")
    print("M9 is disabled by default; M3 remains at weight 0.07.")
    base.main()


if __name__ == "__main__":
    main()
