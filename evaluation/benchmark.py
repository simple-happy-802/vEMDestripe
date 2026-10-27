#!/usr/bin/env python3
"""Publication-oriented benchmark for EM Z-stripe restoration.

This replaces the old hard-coded comparison script. It evaluates Original,
EMISAC, All-In-One-Deflicker, FECNet, vEMDestripe, and vEMDestripe +
Perceptual; handles FECNet TIFF parts; uses 6x Z resampling only for the
explicitly listed anisotropic dataset families in XZ/YZ appearance metrics and views;
and writes per-dataset tables, curves, figures, and a single review PDF.
"""
from __future__ import annotations

import argparse, csv, json, os, pickle, re, shutil, sys, tempfile, threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy.ndimage import gaussian_filter1d, zoom
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:  # Curve-only CUDA runs do not require full-reference metrics.
    peak_signal_noise_ratio = structural_similarity = None
import tifffile

from .niqe import fit_pristine_model, niqe_score, summary

# Kept inside the project so this benchmark does not alter the active conda env.
VENDOR = Path(__file__).resolve().parent / "_benchmark_vendor"
if VENDOR.is_dir():
    sys.path.append(str(VENDOR))
try:
    from brisque import BRISQUE
    from brisque.models import MODEL_PATH as BRISQUE_MODEL_PATH
    from libsvm import svmutil
    from pypiqe import piqe
except ImportError:
    BRISQUE = BRISQUE_MODEL_PATH = svmutil = piqe = None

METHODS = ("Original", "EMISAC", "AIODeflicker", "FECNet", "SSCOR", "vEMDestripe", "vEMDestripeFull", "vEMDestripePerceptual")
MATCHED_METHOD_BY_SOURCE = {
    "EMISAC": "EMISACOriginalMatched",
    "AIODeflicker": "AIODeflickerOriginalMatched",
    "FECNet": "FECNetOriginalMatched",
    "vEMDestripe": "vEMDestripeOriginalMatched",
    "vEMDestripeFull": "vEMDestripeFullOriginalMatched",
    "vEMDestripePerceptual": "vEMDestripePerceptualOriginalMatched",
}
ORIGINAL_MATCHED_METHODS = tuple(MATCHED_METHOD_BY_SOURCE.values())
VEM_MATCH_SOURCES = ("vEMDestripe", "vEMDestripePerceptual")
LABELS = {"Original": "Original", "EMISAC": "EMISAC", "AIODeflicker": "All-In-One-Deflicker",
          "FECNet": "FECNet", "SSCOR": "SSCOR", "vEMDestripe": "vEMDestripe", "vEMDestripeFull": "vEMDestripe (Full)", "vEMDestripePerceptual": "vEMDestripe + Perceptual",
          "EMISACOriginalMatched": "EMISAC + Global Match",
          "AIODeflickerOriginalMatched": "All-In-One-Deflicker + Global Match",
          "FECNetOriginalMatched": "FECNet + Global Match",
          "vEMDestripeOriginalMatched": "vEMDestripe + Global Match",
          "vEMDestripeFullOriginalMatched": "vEMDestripe (Full) + Global Match",
          "vEMDestripePerceptualOriginalMatched": "vEMDestripe + Perceptual + Global Match"}
PLOT_LABELS = {**LABELS,
               "EMISACOriginalMatched": "EMISAC + Match",
               "AIODeflickerOriginalMatched": "AIO + Match",
               "FECNetOriginalMatched": "FECNet + Match",
               "vEMDestripePerceptual": "vEM + P",
               "vEMDestripeOriginalMatched": "vEM + Match",
               "vEMDestripeFull": "vEM Full",
               "vEMDestripeFullOriginalMatched": "vEM Full + Match",
               "vEMDestripePerceptualOriginalMatched": "vEM + P + Match"}
COLORS = {"Original": "#7F7F7F", "EMISAC": "#0072B2", "AIODeflicker": "#009E73",
          "FECNet": "#E69F00", "SSCOR": "#5E3C99", "vEMDestripe": "#D55E00", "vEMDestripeFull": "#8C564B", "vEMDestripePerceptual": "#CC79A7",
          "EMISACOriginalMatched": "#56B4E9", "AIODeflickerOriginalMatched": "#44AA99", "FECNetOriginalMatched": "#F0E442",
          "vEMDestripeOriginalMatched": "#A65628", "vEMDestripeFullOriginalMatched": "#6B4226", "vEMDestripePerceptualOriginalMatched": "#984EA3"}
# These are deliberately the *only* dataset families whose XZ/YZ planes use
# 6x Z interpolation.  Continuity measures (ASID, local-ASID, ISCC, NISD,
# Z-HF and the two Z profiles) remain at native Z sampling for every dataset.
Z6_DATASET_PREFIXES = (
    "kasthuri11", "bumbarger13", "jrc_dauer-larva_recon-1",
    "snemitrain", "snemitest", "minnie65",
)


def nkey(p: Path) -> List[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def spaced(n: int, count: int) -> List[int]:
    if n <= count: return list(range(n))
    return np.linspace(0, n - 1, count).round().astype(int).tolist()


def to_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3: x = x[..., 0]
    if x.dtype == np.uint8: return x
    x = x.astype(np.float32)
    # Fiji FECNet output is uint16 but stores 8-bit intensities (0..255).
    if np.nanmax(x) <= 255.0: return np.clip(x, 0, 255).astype(np.uint8)
    return np.clip(x / np.iinfo(np.uint16).max * 255.0, 0, 255).astype(np.uint8)


class Stack:
    count: int
    def read(self, index: int) -> np.ndarray: raise NotImplementedError
    def read_many(self, ids: Sequence[int]) -> np.ndarray: return np.stack([self.read(i) for i in ids])


class PNGStack(Stack):
    def __init__(self, paths: Sequence[Path]): self.paths, self.count = list(paths), len(paths)
    def read(self, index: int) -> np.ndarray:
        raw = np.fromfile(str(self.paths[index]), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(f"Cannot read {self.paths[index]}")
        return image


class VolumeTIFF(Stack):
    def __init__(self, path: Path):
        self.path, self.array = path, tifffile.memmap(path, mode="r")
        self.count = int(self.array.shape[0]) if self.array.ndim == 3 else 1
    def read(self, index: int) -> np.ndarray: return to_u8(self.array[index] if self.count > 1 else self.array)
    def read_many(self, ids: Sequence[int]) -> np.ndarray:
        return np.stack([to_u8(x) for x in self.array[list(ids)]])


class PagesTIFF(Stack):
    """One or several ordinary multi-page TIFFs, including FECNet part1/part2."""
    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths); self.files = [tifffile.TiffFile(p) for p in self.paths]
        # tifffile's TiffFile owns one seekable file handle.  Concurrent page
        # reads on the same handle corrupt its seek position and can manifest
        # as bogus TIFF tags or empty images.  Full-reference PSNR/SSIM uses a
        # thread pool, so protect each physical TIFF independently.
        self._locks = [threading.RLock() for _ in self.files]
        self.offsets, total = [], 0
        for f in self.files:
            self.offsets.append(total); total += len(f.pages)
        self.count = total
    def read(self, index: int) -> np.ndarray:
        for file, offset, lock in zip(self.files, self.offsets, self._locks):
            if index < offset + len(file.pages):
                # Access the page explicitly.  Some FECNet exports contain one
                # TIFF series per page, for which TiffFile.asarray(key=slice)
                # may silently return an empty array even though file.pages is
                # complete.
                with lock:
                    return to_u8(file.pages[index - offset].asarray())
        raise IndexError(index)
    def read_many(self, ids: Sequence[int]) -> np.ndarray:
        # Page-wise reads are deliberately used here instead of the tempting
        # ``asarray(key=slice(...))`` fast path.  The latter is not valid for
        # TIFFs whose pages belong to separate series (notably the large
        # FECNet simulated-volume outputs).
        return np.stack([self.read(i) for i in ids])
    def close(self) -> None:
        for file in self.files: file.close()


def pngs(folder: Path) -> List[Path]:
    bad = ("reference", "preview", "comparison", "groundtruth", "montage")
    return sorted([p for p in folder.glob("*.png") if not any(x in p.name.lower() for x in bad)], key=nkey)


def configured_root(project: Path, variable: str, relative_default: str) -> Path:
    """Resolve an optional external result root without hard-coded usernames."""
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else project / relative_default


def image_sequence(root: Path, dataset: str) -> PNGStack:
    """Resolve the same supported per-dataset layouts as batch inference."""
    dataset_root = root / dataset
    for folder in (
        dataset_root,
        dataset_root / "input",
        dataset_root / "striped",
        dataset_root / "test" / "striped",
        dataset_root / "test",
    ):
        paths = pngs(folder)
        if paths:
            return PNGStack(paths)
    return PNGStack([])


def uses_z6_resampling(dataset: str) -> bool:
    """Return True only for the six user-specified anisotropic families."""
    name = dataset.casefold()
    return any(name.startswith(prefix) for prefix in Z6_DATASET_PREFIXES)


def fec_stack(root: Path, dataset: str) -> Optional[Stack]:
    folder = root / dataset
    if not folder.is_dir(): return None
    files = sorted(folder.glob("*restored*fiji*.tif"), key=nkey)
    return PagesTIFF(files) if files else None


def sscor_stack(root: Optional[Path], dataset: str) -> Optional[Stack]:
    """Load the latest valid SSCOR restoration without copying its outputs.

    SSCOR keeps each dataset below ``<root>/<dataset>/<run>/restored``.  The
    current reproducible run is ``seed42_fix1``; the fallback makes the loader
    robust to a future run name while requiring a non-empty restored sequence.
    """
    if root is None:
        return None
    dataset_root = root / dataset
    preferred = dataset_root / "seed42_fix1" / "restored"
    choices = [preferred] if preferred.is_dir() else []
    choices.extend(sorted(dataset_root.glob("*/restored"), key=nkey, reverse=True) if dataset_root.is_dir() else [])
    seen = set()
    for folder in choices:
        if folder in seen:
            continue
        seen.add(folder)
        paths = pngs(folder)
        if paths:
            return PNGStack(paths)
    return None


def emisac_stack(root: Path, dataset: str) -> Optional[Stack]:
    paths = sorted((root / f"output_batch_{dataset}").rglob("normalized_data_final.tif"), key=nkey)
    # The three synthetic kidney benchmarks were generated with the earlier
    # EMISAC batch workflow, whose valid outputs live under the sibling
    # ``output-1`` directory rather than ``output-new``.  Discover that
    # established layout as well, so a method is not silently dropped merely
    # because the result-generation workflow changed its output root.
    if not paths:
        paths = sorted((root.parent / "output-1" / f"output_batch_{dataset}").rglob("normalized_data_final.tif"), key=nkey)
    # Bumbarger EMISAC was run later with the GPU ultra-batch workflow rather
    # than the older ``output-new/output_batch_*`` layout.  Treat this as the
    # same method output so all comparison and paper-figure code can discover
    # it without copying a 2-GB TIFF into the project tree.
    if not paths:
        paths = sorted((root.parent / "output_gpu_batch_ultra-2000" / dataset).rglob("normalized_data_final.tif"), key=nkey)
    # The renewed SNEMItrain EMISAC run writes one merged TIFF instead of the
    # legacy ``normalized_data_final.tif`` filename.  It is an aligned 100-Z
    # stack and should participate in the same matched benchmark.
    if not paths and dataset == "snemitrain":
        paths = sorted((root / "output_batch_snemitrain" / "snemitrain").glob("*_merged.tif"), key=nkey)
    return VolumeTIFF(paths[0]) if paths else None


def stacks_for_dataset(root: Path, dataset: str, sscor_root: Optional[Path] = None) -> Dict[str, Optional[Stack]]:
    test_root = configured_root(root, "VEMDESTRIPE_TEST_DATA_ROOT", "test_data")
    simulation_root = configured_root(root, "VEMDESTRIPE_SIM_DATA_ROOT", "testdata-sim")
    vem_root = configured_root(root, "VEMDESTRIPE_RESULT_ROOT", "results/vemdestripe")
    aio_root = configured_root(root, "AIODEFLICKER_RESULT_ROOT", "results/all-in-one-deflicker")
    emisac_root = Path(os.environ.get("EMISAC_RESULT_ROOT", root.parent / "EMISAC-master" / "output-new"))
    fecnet_root = Path(os.environ.get("FECNET_RESULT_ROOT", root.parent / "FECNet-main" / "EMFECNet" / "output"))
    # The three new kidney simulations are intentionally stored outside
    # ``test_data``: their striped input lives in ``testdata-sim/<name>/striped``.
    # Keep that layout explicit instead of requiring duplicated PNG volumes.
    kidney_simulation = dataset.endswith(("-weak", "-medium", "-strong")) and dataset.startswith("jrc_mus-kidney_")
    if kidney_simulation:
        out: Dict[str, Optional[Stack]] = {
            "Original": PNGStack(pngs(simulation_root / dataset / "striped")),
            "AIODeflicker": PNGStack(pngs(aio_root / dataset)),
            "vEMDestripe": PNGStack(pngs(vem_root / dataset)),
            "vEMDestripeFull": None,
            "vEMDestripePerceptual": None,
            "EMISAC": emisac_stack(emisac_root, dataset),
            "FECNet": fec_stack(fecnet_root, dataset),
            "SSCOR": sscor_stack(sscor_root, dataset),
        }
    else:
        out = {
        "Original": image_sequence(test_root, dataset),
        "AIODeflicker": PNGStack(pngs(aio_root / dataset)),
        "vEMDestripe": PNGStack(pngs(vem_root / dataset)),
        "vEMDestripeFull": None,
        "vEMDestripePerceptual": None,
        "EMISAC": emisac_stack(emisac_root, dataset),
        "FECNet": fec_stack(fecnet_root, dataset),
        "SSCOR": sscor_stack(sscor_root, dataset),
        }
    expected = out["Original"].count
    # test_data may contain metadata-only folders. They are recorded in
    # coverage.csv but are not valid volumes to benchmark.
    if expected == 0:
        return {key: None for key in out}
    for key, value in list(out.items()):
        # Bumbarger EMISAC was produced as the aligned leading 2000-slice
        # prefix of a 2651-slice raw volume.  It is valid only for evaluation
        # windows wholly contained in that prefix; the batch evaluator checks
        # the actual selected window before constructing WindowStack.
        partial_bumbarger_emisac = (
            dataset == "bumbarger138092_8092_4096_4096tile_r03_c03_1024_stack"
            and key == "EMISAC" and value is not None and 0 < value.count < expected
        )
        if value is None or (value.count != expected and not partial_bumbarger_emisac): out[key] = None
    return out


def load(stack: Stack, ids: Sequence[int]) -> np.ndarray:
    out = stack.read_many(ids).astype(np.uint8, copy=False)
    if out.ndim != 3: raise ValueError(f"Expected Z,Y,X volume; got {out.shape}")
    return out


def z_windows(n: int, length: int, windows: int, start: int = 0) -> List[List[int]]:
    """Continuous evaluation windows constrained to the held-out Z tail."""
    if not 0 <= start < n:
        raise ValueError(f"test Z start {start} is outside 0..{n - 1}")
    available = n - start
    length = available if length <= 0 else min(length, available)
    starts = spaced(max(1, available - length + 1), windows)
    return [list(range(start + s, start + s + length)) for s in starts]


def cross_section_at(volume: np.ndarray, axis: str, pos: int, z_ratio: float, border: int) -> np.ndarray:
    """Extract an interior XZ/YZ plane, excluding in-plane padding margins."""
    section = volume[:, pos, :] if axis == "xz" else volume[:, :, pos]
    if border:
        if 2 * border >= section.shape[1]:
            raise ValueError(f"Cross-plane border {border} is too large for width {section.shape[1]}.")
        section = section[:, border:-border]
    if z_ratio != 1.0:
        section = zoom(section.astype(np.float32), (z_ratio, 1.0), order=1)
    return section


def cross_sections(volume: np.ndarray, axis: str, count: int, z_ratio: float, border: int) -> Iterable[np.ndarray]:
    side = volume.shape[1] if axis == "xz" else volume.shape[2]
    if 2 * border >= side:
        raise ValueError(f"Cross-plane border {border} is too large for side {side}.")
    for pos in [border + x for x in spaced(side - 2 * border, count)]:
        yield cross_section_at(volume, axis, pos, z_ratio, border)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    if a.std() < 1e-8 or b.std() < 1e-8: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def native_continuity(stack: Stack, pairs: int, profile_points: int, start: int = 0) -> Dict[str, object]:
    if not 0 <= start < stack.count:
        raise ValueError(f"test Z start {start} is outside the stack")
    starts = [start + x for x in spaced(max(1, stack.count - start - 1), pairs)]
    # These calculations remain in native Z. Interpolating first would
    # mechanically reduce the apparent discontinuity.
    diffs, corrs, normalized_diffs, local_asid, local_contrast = [], [], [], [], []
    for z in starts:
        a, b = stack.read(z), stack.read(z + 1)
        d = float(np.mean(np.abs(a.astype(float) - b.astype(float))))
        diffs.append(d)
        c = min(256, a.shape[0], a.shape[1]); y = (a.shape[0] - c) // 2; x = (a.shape[1] - c) // 2
        corrs.append(safe_corr(a[y:y+c, x:x+c], b[y:y+c, x:x+c]))
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        normalized_diffs.append(d / max(float(np.mean(np.hypot(gx, gy))), 1e-6))
        # Local robust continuity: a global slice mean can hide a local stripe.
        # Blocks are fixed in image coordinates and use median/IQR, which are
        # less affected by membranes and outlying bright structures.
        block = max(32, min(a.shape) // 8)
        for y0 in range(0, a.shape[0] - block + 1, block):
            for x0 in range(0, a.shape[1] - block + 1, block):
                aa, bb = a[y0:y0+block, x0:x0+block].astype(float), b[y0:y0+block, x0:x0+block].astype(float)
                ia = np.subtract(*np.percentile(aa, [75, 25])); ib = np.subtract(*np.percentile(bb, [75, 25]))
                local_asid.append(abs(float(np.median(aa) - np.median(bb))) / max((ia + ib) / 2, 1.0))
                local_contrast.append(abs(np.log(ib + 1.0) - np.log(ia + 1.0)))
    ids = [start + x for x in spaced(stack.count - start, profile_points)]
    brightness, roi = [], []
    for z in ids:
        im = stack.read(z); c = min(256, im.shape[0], im.shape[1]); y = (im.shape[0]-c)//2; x = (im.shape[1]-c)//2
        brightness.append(float(im.mean())); roi.append(float(im[y:y+c, x:x+c].mean()))
    b = np.asarray(brightness); high = b - gaussian_filter1d(b, sigma=3, mode="nearest")
    return {"asid_percent": float(np.mean(diffs) / 255 * 100), "iscc": float(np.nanmean(corrs)),
            "nisd": float(np.nanmean(normalized_diffs)),
            "local_asid": float(np.nanmean(local_asid)), "local_contrast_discontinuity": float(np.nanmean(local_contrast)),
            "z_brightness_hf_percent": float(np.sqrt(np.mean(high * high)) / 255 * 100),
            "profile_z": ids, "brightness": brightness, "roi_intensity": roi}


def tenengrad(images: Sequence[np.ndarray]) -> float:
    values = []
    for im in images:
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
        values.append(float(np.mean(gx * gx + gy * gy)))
    return float(np.mean(values))


def finite_mean(values: Sequence[float]) -> float:
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def directional_stripe_energy(section: np.ndarray) -> float:
    """Fraction of 2-D spectral energy due to horizontal Z-band artefacts.

    In an XZ/YZ plane, a Z-oriented intensity flicker appears as horizontal
    bands. Its Fourier energy is therefore concentrated near the vertical
    frequency axis (large f_z, small in-plane frequency). DC is excluded.
    The caller supplies the same XZ/YZ plane used for visual comparison.
    Constant background planes are excluded: their Fourier energy ratio is
    undefined and must not be mistaken for a stripe score after an affine bias.
    """
    image = section.astype(np.float32)
    if float(np.std(image)) < 1e-6:
        return float("nan")
    image -= cv2.GaussianBlur(image, (0, 0), 16)
    image *= np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1])).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
    fz = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))[None, :]
    valid = (np.abs(fz) > 1.0 / image.shape[0]) | (np.abs(fx) > 1.0 / image.shape[1])
    horizontal_band = valid & (np.abs(fz) >= 3.0 * np.abs(fx))
    total = float(power[valid].sum())
    return float(power[horizontal_band].sum() / total) if total > 1e-12 else float("nan")


def make_brisque() -> BRISQUE:
    """Load the published BRISQUE model, including Windows Unicode-path support."""
    if BRISQUE is None or BRISQUE_MODEL_PATH is None or svmutil is None:
        raise RuntimeError("BRISQUE/PIQE dependencies are unavailable. Install _benchmark_vendor packages for the full benchmark.")
    # libsvm cannot open a model whose path contains the Chinese workspace name.
    # The image itself is not copied or transformed; only the static SVM file is.
    model_copy = Path(tempfile.gettempdir()) / "vEMDestripe_brisque_svm.txt"
    shutil.copyfile(Path(BRISQUE_MODEL_PATH) / "svm.txt", model_copy)
    # Avoid BRISQUE.__init__: libsvm emits a spurious error before we can
    # replace its Unicode-path model. These are the exact same initial fields.
    scorer = BRISQUE.__new__(BRISQUE)
    scorer.url = False
    scorer.model = svmutil.svm_load_model(str(model_copy))
    with (Path(BRISQUE_MODEL_PATH) / "normalize.pickle").open("rb") as handle:
        scorer.scale_params = pickle.load(handle)
    return scorer


def standard_iqa(images: Sequence[np.ndarray], scorer: BRISQUE) -> Dict[str, float]:
    """Standard BRISQUE/PIQE on the original, unnormalised uint8 XY pixels."""
    brisque_values, piqe_values = [], []
    for image in images:
        # BRISQUE implementation expects RGB; three identical channels preserve
        # the original grayscale values exactly. PIQE accepts 2-D grayscale.
        rgb = np.repeat(image[..., None], 3, axis=2)
        brisque_values.append(float(scorer.score(rgb)))
        piqe_values.append(float(piqe(image)[0]))
    return {"brisque_xy": float(np.mean(brisque_values)), "piqe_xy": float(np.mean(piqe_values))}


def cached_niqe_scores(images: Sequence[np.ndarray], mu: np.ndarray, cov: np.ndarray, patch: int,
                       cache_path: Optional[Path] = None) -> List[float]:
    """Score a sequence while checkpointing completed per-image NIQE values.

    Full held-out volumes can require more than an interactive process time limit.
    The cache is deliberately limited to deterministic scalar NIQE scores: it does
    not cache images or alter any score, and a rerun simply resumes at the first
    missing image.
    """
    scores: List[float] = []
    if cache_path and cache_path.exists():
        try:
            scores = [float(x) for x in np.load(cache_path)["scores"]]
        except (OSError, KeyError, ValueError):
            scores = []
    scores = scores[:len(images)]
    for index in range(len(scores), len(images)):
        scores.append(float(niqe_score(images[index], mu, cov, patch)))
        if cache_path and (index + 1) % 25 == 0:
            np.savez_compressed(cache_path, scores=np.asarray(scores, dtype=np.float64))
    if cache_path:
        np.savez_compressed(cache_path, scores=np.asarray(scores, dtype=np.float64))
    return scores


def nonreference(stack: Stack, mu: np.ndarray, cov: np.ndarray, patch: int, zratio: float,
                 zlength: int, nwindows: int, nsections: int, iqa_slices: int, scorer: BRISQUE,
                 cross_plane_border: int, test_start: int = 0, xy_test_slices: int = 0,
                 xy_niqe_cache: Optional[Path] = None, xy_test_seed: int = 20260721) -> Tuple[Dict[str, float], Dict[str, object]]:
    xy, xz, yz, sharp, stripe_xz, stripe_yz, xy_z_indices = [], [], [], [], [], [], []
    for ids in z_windows(stack.count, zlength, nwindows, test_start):
        vol = load(stack, ids)
        if xy_test_slices <= 0:
            xy_ids = list(range(len(ids)))
        else:
            # Reproducible random held-out XY selection.  A fixed seed is more
            # appropriate than an unrecorded random draw in a paper benchmark.
            rng = np.random.default_rng(xy_test_seed + ids[0])
            xy_ids = sorted(rng.choice(len(ids), size=min(xy_test_slices, len(ids)), replace=False).tolist())
        images = [vol[i] for i in xy_ids]
        xy_z_indices.extend(int(ids[i]) for i in xy_ids)
        xy += cached_niqe_scores(images, mu, cov, patch, xy_niqe_cache)
        xz += [niqe_score(s, mu, cov, patch) for s in cross_sections(vol, "xz", nsections, zratio, cross_plane_border)]
        yz += [niqe_score(s, mu, cov, patch) for s in cross_sections(vol, "yz", nsections, zratio, cross_plane_border)]
        # User-facing cross-plane assessment is always performed after the
        # same Z upsampling used for XZ/YZ visualization and NIQE.
        stripe_xz += [directional_stripe_energy(s) for s in cross_sections(vol, "xz", nsections, zratio, cross_plane_border)]
        stripe_yz += [directional_stripe_energy(s) for s in cross_sections(vol, "yz", nsections, zratio, cross_plane_border)]
        sharp.append(tenengrad(images))
    continuity = native_continuity(stack, 32, 256, test_start)
    iq_ids = [test_start + x for x in spaced(stack.count - test_start, min(iqa_slices, stack.count - test_start))]
    iqa_metrics = standard_iqa([stack.read(i) for i in iq_ids], scorer)
    result = {"niqe_xy": finite_mean(xy), "niqe_xz": finite_mean(xz), "niqe_yz": finite_mean(yz),
              "stripe_energy_xz": finite_mean(stripe_xz), "stripe_energy_yz": finite_mean(stripe_yz),
              "xy_tenengrad": float(np.mean(sharp)), **iqa_metrics,
              **{k: float(v) for k,v in continuity.items() if isinstance(v, (int,float))}}
    continuity.update({"niqe_xy_samples": [float(v) for v in xy if np.isfinite(v)],
                       "niqe_xz_samples": [float(v) for v in xz if np.isfinite(v)],
                       "niqe_yz_samples": [float(v) for v in yz if np.isfinite(v)],
                       "niqe_xy_z_indices": xy_z_indices})
    return result, continuity


def ground_truth(root: Path, dataset: str) -> Optional[Stack]:
    test_root = configured_root(root, "VEMDESTRIPE_TEST_DATA_ROOT", "test_data")
    for folder in (
        test_root / dataset / "groundtruth",
        test_root / dataset / "test" / "clean",
    ):
        paths = pngs(folder)
        if paths:
            return PNGStack(paths)
    data = configured_root(root, "VEMDESTRIPE_GT_ROOT", "data/ground_truth")
    kidney_clean = "jrc_mus-kidney_em-fibsem-uint8-s0_uint8_x10000-11024_y5000-6024_z5000-6600.tif"
    names = {"feijunyun": kidney_clean,
             "jrc_mus-kidney_em-fibsem-uint8-s0_uint8_x10000-11024_y5000-6024_z5000-6600-weak": kidney_clean,
             "jrc_mus-kidney_em-fibsem-uint8-s0_uint8_x10000-11024_y5000-6024_z5000-6600-medium": kidney_clean,
             "jrc_mus-kidney_em-fibsem-uint8-s0_uint8_x10000-11024_y5000-6024_z5000-6600-strong": kidney_clean,
             "SIM-jrc_mus-liver-3": "jrc_mus-liver-3_recon-1_em_fibsem-uint8_s0_x8000-9024_y8000-9024_z8000-9600_display_norm_110-189_fast.tif"}
    p = data / names[dataset] if dataset in names else None
    return VolumeTIFF(p) if p and p.is_file() else None


def full_reference(pred: Stack, gt: Stack, samples: int = 24, start: int = 0) -> Dict[str, float]:
    if pred.count != gt.count: return {"psnr": np.nan, "ssim": np.nan, "mae": np.nan, "msssim": np.nan, "edge_corr": np.nan}
    psnr, ssim, mae, msssim, edge_corr = [], [], [], [], []
    for z in [start + x for x in spaced(pred.count - start, samples)]:
        a, b = pred.read(z), gt.read(z)
        psnr.append(float("inf") if np.array_equal(a, b) else peak_signal_noise_ratio(b, a, data_range=255))
        ssim.append(structural_similarity(b, a, data_range=255)); mae.append(float(np.mean(abs(a.astype(float)-b.astype(float)))))
        # Five-scale SSIM approximation with the standard coarser-scale emphasis.
        aa, bb, scales, weights = a, b, [], (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
        for _ in weights:
            scales.append(structural_similarity(bb, aa, data_range=255))
            if min(aa.shape) < 32: break
            aa, bb = cv2.pyrDown(aa), cv2.pyrDown(bb)
        w = np.asarray(weights[:len(scales)]); w /= w.sum(); msssim.append(float(np.prod(np.maximum(scales, 1e-8) ** w)))
        ga = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
        gb = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
        edge_corr.append(safe_corr(ga, gb))
    # A few unmodified frames can be exactly equal to GT (infinite PSNR), but
    # an arithmetic mean of inf would conceal the error on all other frames.
    finite_psnr = [v for v in psnr if np.isfinite(v)]
    mean_psnr = float(np.mean(finite_psnr)) if finite_psnr else float("inf")
    return {"psnr": mean_psnr, "ssim": float(np.mean(ssim)), "mae": float(np.mean(mae)),
            "msssim": float(np.mean(msssim)), "edge_corr": float(np.nanmean(edge_corr))}


def style() -> None:
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","DejaVu Serif"], "font.size":8,
                         "axes.titleweight":"bold", "figure.dpi":300, "savefig.dpi":300, "savefig.bbox":"tight"})


def comparison_page(dataset: str, stacks: Dict[str, Optional[Stack]], methods: Sequence[str], zratio: float,
                    cross_plane_border: int, out: Path, pdf: PdfPages, display_mode: str, test_start: int = 0,
                    tag: str = "", title_suffix: str = "") -> None:
    available = [m for m in methods if stacks[m] is not None]; raw = stacks["Original"]; start = max(test_start, raw.count//2-48); ids=list(range(start,min(raw.count,start+96)))
    rvol=load(raw,ids); vmin,vmax=(0,255) if display_mode == "raw" else np.percentile(rvol[len(rvol)//2],(1,99)); style()
    fig, axes=plt.subplots(3,len(available),figsize=(max(11.69,1.95*len(available)),6.4),constrained_layout=True)
    axes=np.atleast_2d(axes)
    for col,m in enumerate(available):
        vol=load(stacks[m],ids); planes=[vol[len(vol)//2], cross_section_at(vol,"xz",vol.shape[1]//2,zratio,cross_plane_border), cross_section_at(vol,"yz",vol.shape[2]//2,zratio,cross_plane_border)]
        for row,(name,im) in enumerate(zip(("XY","XZ","YZ"),planes)):
            ax=axes[row,col]; ax.imshow(im,cmap="gray",vmin=vmin,vmax=vmax,aspect="auto"); ax.set_xticks([]);ax.set_yticks([])
            for s in ax.spines.values(): s.set_linewidth(0.8)
            if col==0: ax.set_ylabel(name,fontweight="bold",fontsize=11)
        axes[0,col].set_title(PLOT_LABELS[m],color=COLORS[m],fontweight="bold",fontsize=8.5)
    display_note="raw 0--255 range" if display_mode == "raw" else "Original 1st--99th percentile range"
    fig.suptitle(f"{dataset}: {title_suffix} ({display_note}; XZ/YZ Z-resampled {zratio:g}x; {cross_plane_border}px in-plane border excluded)",fontweight="bold")
    fig.savefig(out/f"comparison_{dataset}{tag}.png"); fig.savefig(out/f"comparison_{dataset}{tag}.pdf"); pdf.savefig(fig); plt.close(fig)


def cross_section_validation_page(dataset: str, stacks: Dict[str, Optional[Stack]], zratio: float, border: int,
                                  test_start: int, out: Path, pdf: PdfPages, display_mode: str) -> None:
    """Show raw and interior cross-sections, making output-boundary handling auditable."""
    methods = [m for m in ("Original", "vEMDestripe", "vEMDestripePerceptual") if stacks.get(m) is not None]
    raw = stacks["Original"]; start = max(test_start, raw.count//2-48); ids = list(range(start, min(raw.count, start+96)))
    rvol=load(raw,ids); vmin,vmax=(0,255) if display_mode == "raw" else np.percentile(rvol[len(rvol)//2],(1,99)); style()
    fig, axes = plt.subplots(5, len(methods), figsize=(8.8, 9.2), constrained_layout=True); axes=np.atleast_2d(axes)
    names=("XY", "XZ full", f"XZ interior ({border}px)", "YZ full", f"YZ interior ({border}px)")
    for col, method in enumerate(methods):
        vol = load(stacks[method], ids)
        xz_full = vol[:,vol.shape[1]//2,:]; yz_full = vol[:,:,vol.shape[2]//2]
        planes=(vol[len(vol)//2], xz_full, cross_section_at(vol,"xz",vol.shape[1]//2,zratio,border), yz_full, cross_section_at(vol,"yz",vol.shape[2]//2,zratio,border))
        for row,(name,image) in enumerate(zip(names,planes)):
            axes[row,col].imshow(image,cmap="gray",vmin=vmin,vmax=vmax,aspect="auto"); axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if col==0: axes[row,col].set_ylabel(name,fontweight="bold",fontsize=9)
        axes[0,col].set_title(PLOT_LABELS[method],color=COLORS[method],fontweight="bold",fontsize=9)
    fig.suptitle(f"{dataset}: held-out Z cross-section validation (full vs interior crop)",fontweight="bold")
    fig.savefig(out/f"cross_section_validation_{dataset}.png"); fig.savefig(out/f"cross_section_validation_{dataset}.pdf"); pdf.savefig(fig); plt.close(fig)


def curves_page(dataset: str, profiles: Dict[str, Dict[str, object]], methods: Sequence[str], out: Path, pdf: PdfPages, tag: str = "") -> None:
    style(); fig,axes=plt.subplots(2,1,figsize=(11.69,6.4),sharex=True,constrained_layout=True)
    for m in methods:
        if m not in profiles: continue
        p=profiles[m]
        z=np.asarray(p["profile_z"]); axes[0].plot(z,p["brightness"],label=PLOT_LABELS[m],color=COLORS[m],lw=1.2)
        axes[1].plot(z,p["roi_intensity"],label=PLOT_LABELS[m],color=COLORS[m],lw=1.2)
    axes[0].set_ylabel("Global slice brightness"); axes[1].set_ylabel("Central-ROI intensity"); axes[1].set_xlabel("Native Z-slice index")
    axes[0].set_title("Z-axis brightness profile",fontweight="bold"); axes[1].set_title("Z-axis local intensity profile",fontweight="bold")
    for ax in axes: ax.grid(alpha=.2); ax.legend(ncol=3,fontsize=7)
    fig.suptitle(f"{dataset}: native-Z profiles (no interpolation)",fontweight="bold")
    fig.savefig(out/f"profiles_{dataset}{tag}.png"); fig.savefig(out/f"profiles_{dataset}{tag}.pdf"); pdf.savefig(fig); plt.close(fig)


def niqe_distribution_page(dataset: str, profiles: Dict[str, Dict[str, object]], methods: Sequence[str], out: Path,
                           pdf: PdfPages, tag: str, title_suffix: str) -> List[Dict[str, object]]:
    """Histogram-as-line NIQE distributions, following the requested paper-style plot."""
    style(); fig, axes = plt.subplots(1, 3, figsize=(11.69, 3.35), constrained_layout=True)
    samples: List[Dict[str, object]] = []
    for ax, key, title in zip(axes, ("niqe_xz_samples","niqe_yz_samples","niqe_xy_samples"), ("X-Z cross-sections quality", "Y-Z cross-sections quality", "X-Y slices quality")):
        values = {m:np.asarray(profiles[m].get(key,[]),dtype=float) for m in methods if m in profiles}
        parts = [x[np.isfinite(x)] for x in values.values() if np.any(np.isfinite(x))]
        if not parts: continue
        finite = np.concatenate(parts)
        upper=max(float(np.percentile(finite,99.5)), 1.0); bins=np.linspace(0,upper,41); centers=(bins[:-1]+bins[1:])/2
        for m, x in values.items():
            x=x[np.isfinite(x)]
            if not len(x): continue
            counts,_=np.histogram(np.clip(x,0,upper),bins=bins)
            ax.plot(centers,counts,drawstyle="steps-mid",label=PLOT_LABELS[m],color=COLORS[m],lw=1.25)
            samples.extend({"dataset":dataset,"group":title_suffix,"method":m,"plane":key.replace("niqe_","").replace("_samples",""),"niqe_score":float(v)} for v in x)
        ax.set_title(title,fontweight="bold"); ax.set_xlabel("NIQE score"); ax.set_ylabel("Count"); ax.grid(alpha=.2); ax.legend(fontsize=6.4,ncol=1,loc="upper right")
    fig.suptitle(f"{dataset}: held-out NIQE score distributions - {title_suffix}",fontweight="bold")
    fig.savefig(out/f"niqe_distribution_{dataset}{tag}.png"); fig.savefig(out/f"niqe_distribution_{dataset}{tag}.pdf"); pdf.savefig(fig); plt.close(fig)
    return samples


def table_page(dataset: str, rows: List[Dict[str, object]], out: Path, pdf: PdfPages) -> None:
    style(); keys=["niqe_xy","niqe_xz","niqe_yz","brisque_xy","piqe_xy","asid_percent","iscc","nisd","z_brightness_hf_percent","xy_tenengrad","psnr","ssim","mae"]
    labels=["NIQE-XY↓","NIQE-XZ↓","NIQE-YZ↓","ASID%↓","ISCC↑","Z-HF%↓","Tenengrad↑","PSNR↑","SSIM↑","MAE↓"]
    labels=["NIQE-XY down","NIQE-XZ down","NIQE-YZ down","BRISQUE down","PIQE down","ASID% down","ISCC up","NISD down","Z-HF% down","Tenengrad up","PSNR up","SSIM up","MAE down"]
    body=[]
    for row in rows:
        values=[]
        for k in keys:
            value=row.get(k,np.nan)
            values.append("infinity" if np.isposinf(value) else ("-" if not np.isfinite(value) else f"{value:.3f}"))
        body.append([PLOT_LABELS[row["method"]]]+values)
    widths=[.15]+[.065]*len(labels)
    fig,ax=plt.subplots(figsize=(13.8,max(3.2,.34*(len(body)+1))));ax.axis("off"); tab=ax.table(cellText=body,colLabels=["Method"]+labels,loc="center",cellLoc="center",colWidths=widths)
    tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,1.6)
    for j in range(len(labels)+1): tab[(0,j)].set_text_props(weight="bold")
    for i,row in enumerate(rows,1): tab[(i,0)].set_text_props(weight="bold",color=COLORS[row["method"]])
    # Best finite score in each column: lower is better except correlation,
    # sharpness, PSNR and SSIM. Original is intentionally included because it
    # is an experimental baseline, not a decoration.
    higher_is_better={"iscc","xy_tenengrad","psnr","ssim"}
    for column,key in enumerate(keys,1):
        values=np.asarray([row.get(key,np.nan) for row in rows],dtype=float)
        finite=np.isfinite(values)
        if not np.any(finite): continue
        best=np.max(values[finite]) if key in higher_is_better else np.min(values[finite])
        for i,value in enumerate(values,1):
            if np.isfinite(value) and np.isclose(value,best,rtol=1e-8,atol=1e-10): tab[(i,column)].set_text_props(weight="bold")
    ax.set_title(f"{dataset}: objective metrics (best per column in bold)",fontweight="bold",pad=14); fig.savefig(out/f"metrics_{dataset}.png"); fig.savefig(out/f"metrics_{dataset}.pdf"); pdf.savefig(fig); plt.close(fig)


def specialized_table_page(dataset: str, rows: List[Dict[str, object]], out: Path, pdf: PdfPages) -> None:
    """Stripe, local-continuity and full-reference safeguards kept readable."""
    style(); keys=["stripe_energy_xz","stripe_energy_yz","local_asid","local_contrast_discontinuity","msssim","edge_corr"]
    labels=["Stripe-XZ down","Stripe-YZ down","Local-ASID down","Local contrast jump down","MS-SSIM up","Edge corr. up"]
    body=[]
    for row in rows:
        body.append([PLOT_LABELS[row["method"]]]+["-" if not np.isfinite(row.get(k,np.nan)) else f"{row[k]:.4f}" for k in keys])
    fig,ax=plt.subplots(figsize=(11.69,max(3.2,.34*(len(body)+1))));ax.axis("off");tab=ax.table(cellText=body,colLabels=["Method"]+labels,loc="center",cellLoc="center",colWidths=[.22]+[.13]*len(labels))
    tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.6)
    for j in range(len(labels)+1): tab[(0,j)].set_text_props(weight="bold")
    for i,row in enumerate(rows,1): tab[(i,0)].set_text_props(weight="bold",color=COLORS[row["method"]])
    for col,key in enumerate(keys,1):
        values=np.asarray([row.get(key,np.nan) for row in rows],dtype=float); finite=np.isfinite(values)
        if np.any(finite):
            best=np.max(values[finite]) if key in {"msssim","edge_corr"} else np.min(values[finite])
            for i,value in enumerate(values,1):
                if np.isfinite(value) and np.isclose(value,best,rtol=1e-8,atol=1e-10): tab[(i,col)].set_text_props(weight="bold")
    ax.set_title(f"{dataset}: stripe / local-structure metrics (best per column in bold)",fontweight="bold",pad=14);fig.savefig(out/f"metrics_specialized_{dataset}.png");fig.savefig(out/f"metrics_specialized_{dataset}.pdf");pdf.savefig(fig);plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys=sorted({k for r in rows for k in r});
    with path.open("w",newline="",encoding="utf8") as f: w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def protocol_page(pdf: PdfPages, niqe_protocol: str, cross_plane_border: int) -> None:
    """Chinese protocol page, including the mandatory Z-resampling rule."""
    style(); cn = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis("off")
    text = (
        "评估原则\n\n"
        "1. 所有方法直接读取文件中存储的原始灰度；不做百分位截断、直方图匹配、min-max 或对比度归一化。展示图固定采用 0--255 灰度范围。\n\n"
        f"2. XZ/YZ 评估严格由连续 Z 窗口构成，并统一排除每个截面左右各 {cross_plane_border} 个像素的图像边界，以避免网络零填充影响指标。对于 Kasthuri、bumbarger13、dauer、SNEMI 和 Minnie 等各向异性体数据，先在 Z 方向作 6 倍线性上采样，随后才计算 XZ/YZ-NIQE 与方向频域条纹能量，并生成截面对比图。\n\n"
        "3. ASID、局部 ASID、局部对比度跳变、ISCC、NISD 和 Z-HF 保持原始相邻层采样；它们刻画真实相邻切片的跳变，不能先插值，否则会人为降低不连续性。\n\n"
        "4. feijunyun 与 SIM-jrc_mus-liver-3 具有干净参考体数据，因此 PSNR、SSIM、MS-SSIM、MAE 和边缘相关性是主要证据。无参考指标只能作为补充，不能单独证明结构真实。\n\n"
        f"NIQE 先验：{niqe_protocol}"
    )
    ax.text(.055,.94,text,va="top",ha="left",fontsize=12,fontproperties=cn,wrap=True,linespacing=1.65)
    ax.set_title("vEMDestripe 体电镜去条纹评估协议",fontproperties=cn,fontweight="bold",fontsize=19,pad=22); pdf.savefig(fig); plt.close(fig)


def metric_dictionary_page(pdf: PdfPages) -> None:
    """Chinese, paper-ready definitions of every metric emitted by this program."""
    style(); cn = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis("off")
    text = (
        "指标中文说明（↑ 越大越好；↓ 越小越好）\n\n"
        "• PSNR、SSIM、MS-SSIM、MAE：仅模拟集使用，衡量输出与干净真值的像素、单尺度结构、多尺度结构及绝对误差。\n"
        "• Edge corr.：输出和真值 Sobel 梯度幅值的相关性，用于检验膜边缘与细胞器轮廓是否被保留。\n"
        "• Stripe-XZ / Stripe-YZ：上采样后的 XZ/YZ 截面中，沿 Z 方向形成水平条带的方向性 Fourier 能量比例；用于量化残余条纹。\n"
        "• ASID%：全图相邻层平均绝对强度变化；反映全局闪烁。\n"
        "• Local-ASID：固定块内相邻层中位亮度跳变，以局部 IQR 归一；防止全局均值掩盖局部条纹。\n"
        "• Local contrast jump：固定块 IQR 对比度在相邻层之间的对数变化；检验增强后局部对比度是否随 Z 稳定。\n"
        "• ISCC：中心 ROI 的相邻层相关系数；相邻结构连续时更高。\n"
        "• NISD：相邻层绝对差除以面内梯度；减少“原图纹理丰富导致差异大”的混杂。\n"
        "• Z-HF%：全局层亮度曲线去趋势后的高频 RMS；反映闪烁残余。\n"
        "• Tenengrad：XY 图像 Sobel 梯度能量；作为锐度/细节保护的防护指标。\n"
        "• NIQE、BRISQUE、PIQE：无参考自然图像质量代理。它们可作辅助横向比较，但未针对 EM 训练，不能替代真值、边缘或下游分割验证。"
    )
    ax.text(.055,.94,text,va="top",ha="left",fontsize=11,fontproperties=cn,wrap=True,linespacing=1.65)
    ax.set_title("指标定义与解释",fontproperties=cn,fontweight="bold",fontsize=19,pad=22); pdf.savefig(fig); plt.close(fig)


def main():
    """Compatibility entry point; evaluate saved final files unchanged."""
    from .evaluate import main as evaluate_saved_outputs
    evaluate_saved_outputs()


if __name__ == '__main__': main()
