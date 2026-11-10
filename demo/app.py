"""Small-volume browser demo for the fixed vEMDestripe inference workflow."""
from __future__ import annotations

from io import BytesIO
import logging
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter, time
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from PIL import Image

MAX_ZIP_BYTES = 100 * 2**20
MAX_SECTION_BYTES = 2 * 2**20
MAX_SECTIONS = 128
MAX_SIDE = 512
RETENTION_SECONDS = 3600
AXIS_INDEX = {"XY": 0, "XZ": 1, "YZ": 2}


def extract_sections(archive, destination):
    """Validate and unpack a small PNG stack without extracting archive paths."""
    archive = Path(archive)
    destination = Path(destination)
    if archive.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("The ZIP exceeds 100 MiB.")
    seen = set()
    shape = None
    with ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > 2 * MAX_SECTIONS:
            raise ValueError("The ZIP contains too many entries.")
        sections = []
        for item in entries:
            name = item.orig_filename
            parts = name.rstrip("/").split("/")
            if (
                name != item.filename
                or any(part in {"", ".", ".."} for part in parts)
                or any(ord(char) < 32 or char in '<>:"\\|?*' for char in name)
                or any(part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
                       for part in parts)
            ):
                raise ValueError("The ZIP contains an unsupported path.")
            mode = stat.S_IFMT(item.external_attr >> 16)
            if mode not in {0, stat.S_IFREG, stat.S_IFDIR} or item.flag_bits & 1:
                raise ValueError("Links, special files and encrypted entries are unsupported.")
            if item.is_dir():
                continue
            basename = parts[-1]
            if parts[0] == "__MACOSX" or basename in {".DS_Store", "Thumbs.db"}:
                continue
            if Path(basename).suffix.lower() != ".png":
                raise ValueError("The ZIP should contain one volume of grayscale PNG sections.")
            key = basename.casefold()
            if key in seen:
                raise ValueError("Each section needs a unique filename.")
            seen.add(key)
            if item.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
                raise ValueError("Use a standard ZIP archive.")
            if not 0 < item.file_size <= MAX_SECTION_BYTES:
                raise ValueError("Each PNG file must be at most 2 MiB.")
            sections.append((item, basename))
        if not 16 <= len(sections) <= MAX_SECTIONS:
            raise ValueError("Provide between 16 and 128 PNG sections.")
        if len({item.filename.rpartition("/")[0] for item, _ in sections}) != 1:
            raise ValueError("Put all sections in the same folder within the ZIP.")
        if sum(item.file_size for item, _ in sections) > MAX_ZIP_BYTES:
            raise ValueError("The unpacked PNG files exceed 100 MiB.")
        destination.mkdir()
        for item, basename in sections:
            with source.open(item) as stream:
                data = stream.read(MAX_SECTION_BYTES + 1)
            if len(data) != item.file_size:
                raise ValueError("A ZIP entry has an invalid size.")
            if not data.startswith(b"\x89PNG\r\n\x1a\n") or data[24:26] != b"\x08\x00":
                raise ValueError(f"Expected an 8-bit grayscale PNG: {basename}")
            with Image.open(BytesIO(data), formats=["PNG"]) as image:
                if any(side < 32 or side > MAX_SIDE or side % 8 for side in image.size):
                    raise ValueError("Width and height must be 32–512 pixels and divisible by 8.")
                if image.mode != "L" or getattr(image, "is_animated", False):
                    raise ValueError(f"Expected a single grayscale section: {basename}")
                if shape is None:
                    shape = image.size
                elif image.size != shape:
                    raise ValueError("All sections must have the same width and height.")
                image.load()
            (destination / basename).write_bytes(data)


def process_archive(archive, job, checkpoint, device, progress=None):
    from vemdestripe.optimized import read_section, run_volume

    started = perf_counter()
    inputs = job / "input"
    output = job / "output"
    extract_sections(archive, inputs)
    record = run_volume(inputs, output, checkpoint, device, progress=progress)
    # Disk-backed stacks keep section viewing independent of inference and of
    # the size of the browser session state.
    for folder, stem in ((inputs, "original"), (output / "restored", "restored")):
        stack = np.lib.format.open_memmap(job / f"{stem}.npy", mode="w+", dtype=np.uint8,
                                          shape=tuple(record["shape_zyx"]))
        try:
            for z, name in enumerate(record["input_files"]):
                stack[z] = read_section(folder / name)[0]
            stack.flush()
        finally:
            del stack

    download = job / "vEMDestripe-results.zip"
    with ZipFile(download, "w", compression=ZIP_STORED) as result:
        result.write(output / "processing.json", "processing.json")
        for name in record["input_files"]:
            result.write(output / "restored" / name, f"restored/{name}")
    return record, download, perf_counter() - started


def section_views(job, plane, index):
    axis = AXIS_INDEX[plane]
    views = []
    for stem in ("original", "restored"):
        stack = np.load(job / f"{stem}.npy", mmap_mode="r", allow_pickle=False)
        try:
            pixels = np.take(stack, index, axis=axis)
        finally:
            del stack
        views.append(Image.fromarray(pixels))
    return views


def build_app(checkpoint, device="cpu"):
    import gradio as gr

    # The retained temporary directory belongs only to this app instance.
    workspace = TemporaryDirectory(prefix="vemdestripe-demo-")
    root = Path(workspace.name).resolve()

    def cleanup(state):
        if state is not None:
            job = Path(state["job"]).resolve()
            if job.parent == root and job.name.startswith("job-") and job.is_dir():
                shutil.rmtree(job)

    def images(state, plane, index):
        if state is None:
            return None, None
        job = Path(state["job"])
        try:
            index = max(0, min(int(index), state["shape"][AXIS_INDEX[plane]] - 1))
            return section_views(job, plane, index)
        except FileNotFoundError:
            raise gr.Error("This result has expired. Upload the ZIP again.") from None

    def change_plane(state, plane):
        if state is None:
            return gr.update(), None, None
        end = state["shape"][AXIS_INDEX[plane]] - 1
        index = end // 2
        before, after = images(state, plane, index)
        return gr.update(maximum=end, value=index, interactive=True), before, after

    def run(archive, previous, progress=gr.Progress()):
        if not archive:
            raise gr.Error("Upload a ZIP of grayscale PNG sections first.")
        # Also reclaim jobs abandoned by clients during an upload or callback.
        for folder in root.glob("job-*"):
            if time() - folder.stat().st_mtime > RETENTION_SECONDS:
                cleanup({"job": str(folder)})
        job = Path(mkdtemp(prefix="job-", dir=workspace.name))
        state = {"job": str(job)}
        try:
            progress(0, desc="Checking the ZIP")
            record, download, request_seconds = process_archive(
                archive, job, checkpoint, device,
                lambda done, total: progress(0.05 + 0.85 * done / total,
                                             desc=f"Processing section {done}/{total}"),
            )
            state["shape"] = record["shape_zyx"]
            end = state["shape"][0] - 1
            index = end // 2
            before, after = images(state, "XY", index)
        except (ValueError, BadZipFile, Image.DecompressionBombError) as exc:
            cleanup(state)
            raise gr.Error(str(exc)) from None
        except Exception:
            cleanup(state)
            logging.exception("vEMDestripe demo failed")
            raise gr.Error("The volume could not be processed. Check the ZIP and try again.") from None
        cleanup(previous)
        z, y, x = state["shape"]
        summary = (
            f"**{z} sections · {x} × {y} px**  \n"
            f"Processing **{record['processing_seconds']:.2f} s**  \n"
            f"Server total **{request_seconds:.2f} s** including ZIP handling and export"
        )
        return (state, gr.update(value="XY"), gr.update(maximum=end, value=index, interactive=True),
                before, after, summary, gr.update(value=str(download), interactive=True))

    with gr.Blocks(title="vEMDestripe · Volume restoration", delete_cache=(600, RETENTION_SECONDS)) as demo:
        gr.Markdown(
            "# vEMDestripe\n"
            "Restore axial stripes in a small volume electron microscopy stack."
        )
        state = gr.State(time_to_live=RETENTION_SECONDS, delete_callback=cleanup)
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                upload = gr.File(label="Drag a ZIP here", file_types=[".zip"], type="filepath")
                gr.Markdown(
                    "16–128 numbered, grayscale 8-bit PNG sections. "
                    "Same XY dimensions, 32–512 pixels per side, divisible by 8."
                )
                start = gr.Button("Process volume", variant="primary")
                timing = gr.Markdown()
                download = gr.DownloadButton("Download restored PNGs and run record", interactive=False)
            with gr.Column(scale=3, min_width=300):
                with gr.Row():
                    plane = gr.Radio(list(AXIS_INDEX), value="XY", label="Plane")
                    index = gr.Slider(0, 1, value=0, step=1, label="Section index (zero-based)",
                                      interactive=False)
                with gr.Row():
                    original = gr.Image(label="Original", format="png", interactive=False)
                    restored = gr.Image(label="Restored", format="png", interactive=False)
                gr.Markdown("Both views use the same 8-bit grayscale range and native pixels.")
        start.click(run, [upload, state], [state, plane, index, original, restored, timing, download],
                    concurrency_limit=1, api_name="restore")
        plane.input(change_plane, [state, plane], [index, original, restored], api_name="plane")
        index.release(images, [state, plane, index], [original, restored], api_name="section")
    return demo.queue(max_size=4, default_concurrency_limit=1)


def main():
    import cv2
    import torch

    from vemdestripe.optimized import DEFAULT_CHECKPOINT

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = os.environ.get("VEM_DEVICE", "cpu")
    checkpoint = Path(os.environ.get("VEM_CHECKPOINT", DEFAULT_CHECKPOINT)).resolve(strict=True)
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Set VEM_DEVICE to cpu or to cuda on a CUDA-enabled host.")
    build_app(checkpoint, device).launch(
        server_name=os.environ.get("VEM_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("VEM_PORT", "7860")),
        share=False, max_file_size=MAX_ZIP_BYTES, enable_monitoring=False, run_history=False,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
