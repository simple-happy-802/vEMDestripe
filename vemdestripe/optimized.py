"""Fixed E1 inference and volume postprocessing for native grayscale PNGs."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import mmap
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
from natsort import natsorted
import numpy as np
import torch

from .model import create_model
from .optimized_alignment import fit_volume_luts
from .reference import build_reference


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "vemdestripe_e1_block32" / "best.pth"
)
CHECKPOINT_SHA256 = "9dc1c35debd78256c35419fb36dd2111c4176a0b9da30f3bdabf537f43d82321"


def ordered_map(pool, function, items, limit):
    """Map in input order with a bounded number of pending sections."""
    iterator = iter(items)
    pending = deque()
    for item in iterator:
        pending.append(pool.submit(function, item))
        if len(pending) == limit:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def read_section(path):
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Expected a PNG file: {path.name}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit grayscale PNG: {path.name}")
    return image, hashlib.sha256(data).hexdigest()


def histogram(image):
    if image.size <= 2**24:
        counts = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel().astype(np.float64)
        if counts.sum() != image.size:
            raise ValueError("Histogram lost pixel mass")
    else:
        counts = np.bincount(image.ravel(), minlength=256)
    return counts / image.size


def select_references(histograms):
    levels = np.arange(256)
    means = histograms @ levels
    stds = np.sqrt(np.maximum(0, histograms @ (levels**2) - means**2))
    saturation = histograms[:, [0, 1, 254, 255]].sum(axis=1)
    eligible = np.flatnonzero((stds > 2) & (saturation < 0.4))
    if len(eligible) < 16:
        raise ValueError("At least 16 usable reference sections are required")

    def deviation(values):
        center = np.median(values[eligible])
        distance = np.abs(values - center)
        return distance / max(1, np.median(distance[eligible]))

    rank = deviation(means) + deviation(stds) + 4 * saturation
    return [int(group[np.argmin(rank[group])]) for group in np.array_split(eligible, 16)]


def run_volume(input_dir, output_dir, checkpoint=DEFAULT_CHECKPOINT, device="cpu", workers=4,
               progress=None):
    """Restore a PNG directory and return the completed processing record.

    Each section keeps its input filename. Progress receives completed and total
    inference section counts. A completed output directory has processing.json.
    """
    started = perf_counter()
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Choose an output directory outside the input directory")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if workers < 1:
        raise ValueError("workers must be positive")
    paths = natsorted(
        sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        key=lambda p: p.name,
    )
    if len(paths) < 16:
        raise ValueError("Provide at least 16 grayscale PNG sections")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Choose a CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose CPU or install a CUDA-enabled PyTorch build.")

    checkpoint_bytes = Path(checkpoint).read_bytes()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint checksum differs from the released E1 block32 weights")
    weights = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    model = create_model()
    model.load_state_dict(weights["model_state_dict"], strict=True)
    model.to(device).eval()
    del weights, checkpoint_bytes

    def inspect(path):
        image, digest = read_section(path)
        return image.shape, digest, histogram(image)

    with ThreadPoolExecutor(max_workers=workers) as readers:
        records = list(ordered_map(readers, inspect, paths, workers))
    shape = records[0][0]
    if any(record[0] != shape for record in records):
        raise ValueError("All input sections must have identical XY dimensions")
    if any(size < 32 or size % 8 for size in shape):
        raise ValueError("Each XY dimension must be at least 32 and divisible by 8")
    input_hist = np.stack([record[2] for record in records])
    indices = select_references(input_hist)

    def read_checked(index):
        image, digest = read_section(paths[index])
        if digest != records[index][1]:
            raise ValueError(f"Input changed during processing: {paths[index].name}")
        return image

    def tensor(image):
        return torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)

    reference = tensor(build_reference([read_checked(i) for i in indices], block_size=32))
    output_dir.mkdir(parents=True, exist_ok=False)
    volume_shape = (len(paths), *shape)
    with TemporaryDirectory(prefix=".processing-", dir=output_dir) as scratch:
        scratch = Path(scratch)
        # A disk-backed byte spool keeps the volume out of the Python heap.
        # Use an explicit mmap lifetime so the scratch file can be removed on Windows.
        with (scratch / "network_raw.bin").open("w+b") as stream:
            stream.truncate(int(np.prod(volume_shape)))
            with mmap.mmap(stream.fileno(), 0) as mapping:
                spool = np.ndarray(volume_shape, dtype=np.uint8, buffer=mapping)
                try:
                    source_hist = np.empty_like(input_hist)
                    def store_section(z, raw):
                        spool[z] = raw
                        source_hist[z] = histogram(raw)

                    pending_writes = deque()
                    with (
                        ThreadPoolExecutor(max_workers=workers) as readers,
                        ThreadPoolExecutor(max_workers=1) as spool_writer,
                        torch.no_grad(),
                    ):
                        for z, image in enumerate(
                            ordered_map(readers, read_checked, range(len(paths)), workers)
                        ):
                            prediction = model(torch.cat((tensor(image), reference), dim=1))
                            pixels = prediction[0, 0].cpu().numpy()
                            if not np.isfinite(pixels).all():
                                raise ValueError(f"Nonfinite network output for {paths[z].name}")
                            raw = np.rint(np.clip(pixels * 255.0, 0.0, 255.0)).astype(np.uint8)
                            if len(pending_writes) == 2:
                                pending_writes.popleft().result()
                            pending_writes.append(spool_writer.submit(store_section, z, raw))
                            if progress is not None:
                                progress(z + 1, len(paths))
                        while pending_writes:
                            pending_writes.popleft().result()
                    luts = fit_volume_luts(source_hist, input_hist)
                    restored = scratch / "restored"
                    restored.mkdir()

                    def save_section(z):
                        final = cv2.LUT(spool[z], luts[z])
                        ok, encoded = cv2.imencode(".png", final, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        if not ok:
                            raise OSError(f"PNG encoding failed for {paths[z].name}")
                        (restored / paths[z].name).write_bytes(encoded.tobytes())

                    with ThreadPoolExecutor(max_workers=workers) as writers:
                        for _ in ordered_map(writers, save_section, range(len(paths)), workers):
                            pass
                finally:
                    del spool

        restored.rename(output_dir / "restored")

    record = {
        "status": "complete",
        "workflow": "fixed-e1-block32-v1",
        "checkpoint_sha256": digest,
        "device": str(device),
        "shape_zyx": list(volume_shape),
        "input_files": [path.name for path in paths],
        "input_sha256": {path.name: row[1] for path, row in zip(paths, records)},
        "reference_files": [paths[i].name for i in indices],
        "reference_block_size": 32,
        "postprocessing": {"scheme": "trend_both_s16", "reference": "complete input volume"},
        "processing_seconds": perf_counter() - started,
    }
    marker = output_dir / "processing.json.tmp"
    marker.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marker.replace(output_dir / "processing.json")
    return record
