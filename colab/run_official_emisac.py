#!/usr/bin/env python3
"""Run the authors' official EMISAC source on a directory of image slices.

The upstream EMISAC implementation was published for Python 2 in 2014.  This
launcher fetches one immutable upstream commit and applies a small, auditable
Python-3 compatibility transform.  The numerical objective, bounds, three
optimization stages, smoothing procedure, and default paper parameters are not
reimplemented here.

Official source: https://github.com/azadis/EMISAC
Pinned commit:   7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5

Typical Colab use::

    python run_official_emisac.py run \
        --input-dir demo_data/original \
        --output-dir live_results/EMISAC \
        --work-dir /content/emisac_work \
        --workers 2

The output contains one ``*.png`` for every input slice plus
``run_metadata.json``.  Input basenames are preserved, so a directory holding
``slice_00777.png`` produces ``slice_00777.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence


OFFICIAL_REPOSITORY = "https://github.com/azadis/EMISAC.git"
OFFICIAL_COMMIT = "7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5"
COMPATIBILITY_VERSION = 3
PATCH_MANIFEST = "VEMDESTRIPE_EMISAC_COMPATIBILITY.json"
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


class EmisacCompatibilityError(RuntimeError):
    """Raised when the official checkout cannot be safely prepared or used."""


def _run(command: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in command)
    print(f"[EMISAC official] $ {printable}", flush=True)
    return subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
    )


def _capture(command: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(source_dir: Path) -> str:
    try:
        return _capture(["git", "rev-parse", "HEAD"], cwd=source_dir)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EmisacCompatibilityError(
            f"{source_dir} is not a readable Git checkout: {exc}"
        ) from exc


def _require_clean_checkout(source_dir: Path) -> None:
    """Reject pre-existing tracked edits before compatibility patching."""
    status = _capture(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source_dir
    )
    if status:
        raise EmisacCompatibilityError(
            "Official checkout contains tracked modifications before compatibility "
            "patching; prepare a fresh work directory."
        )


def clone_official_source(source_dir: Path) -> None:
    """Fetch exactly the pinned official commit into a new directory."""
    source_dir = source_dir.resolve()
    if source_dir.exists():
        raise EmisacCompatibilityError(
            f"Refusing to replace existing source directory: {source_dir}"
        )
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", OFFICIAL_REPOSITORY, source_dir])
        _run(["git", "fetch", "--depth", "1", "origin", OFFICIAL_COMMIT], cwd=source_dir)
        _run(["git", "checkout", "--detach", OFFICIAL_COMMIT], cwd=source_dir)
    except Exception:
        # A failed clone is not a valid cache.  Remove only the exact directory
        # created by this function and never a user-provided pre-existing path.
        if source_dir.exists():
            shutil.rmtree(source_dir)
        raise
    actual = _git_commit(source_dir)
    if actual != OFFICIAL_COMMIT:
        raise EmisacCompatibilityError(
            f"Official commit mismatch: expected {OFFICIAL_COMMIT}, found {actual}"
        )


def _replace_exact(text: str, old: str, new: str, *, expected: int = 1) -> str:
    count = text.count(old)
    if count != expected:
        raise EmisacCompatibilityError(
            f"Compatibility patch expected {expected} occurrence(s), found {count}: {old!r}"
        )
    return text.replace(old, new)


def _python3_refactor(files: Iterable[Path]) -> None:
    """Convert the only Python-2 syntax used upstream: print statements.

    Python 3.13 removed :mod:`lib2to3`, so relying on ``2to3`` would make the
    public demo less portable.  The pinned EMISAC files use none of the other
    Python-2-only constructs; an intentionally narrow line transform is easier
    to audit and fails if no expected statements are found.
    """
    converted_total = 0
    print_statement = re.compile(r"^(?P<indent>\s*)print\s+(?P<body>.+?)\s*$")
    for path in files:
        output_lines = []
        converted = 0
        for line in path.read_text(encoding="utf-8").expandtabs(4).splitlines():
            match = print_statement.match(line)
            if match and not match.group("body").startswith("("):
                line = f"{match.group('indent')}print({match.group('body')})"
                converted += 1
            output_lines.append(line)
        if converted == 0:
            raise EmisacCompatibilityError(f"No Python-2 print statements found in {path.name}")
        path.write_text("\n".join(output_lines) + "\n", encoding="utf-8", newline="\n")
        converted_total += converted
    print(f"[EMISAC official] converted {converted_total} Python-2 print statements")


def _patch_main(text: str) -> str:
    text = text.expandtabs(4)
    text = _replace_exact(text, "import skimage.filter.rank as rank", "from skimage.filters import rank")
    text = _replace_exact(text, "import Image", "from PIL import Image")
    text = _replace_exact(
        text,
        "from skimage.morphology import rectangle",
        """try:\n    from skimage.morphology import rectangle\nexcept ImportError:\n    from skimage.morphology import footprint_rectangle\n\n    def rectangle(nrows, ncols):\n        return footprint_rectangle((nrows, ncols))""",
    )
    text = _replace_exact(text, "stack=data[group_name].value", "stack = data[group_name][()]")
    text = _replace_exact(text, "param=f[group_name].value", "param = f[group_name][()]")
    text = text.replace("np.float(", "float(")
    text = text.replace("rank.bilateral_mean(", "rank.mean_bilateral(")
    text = text.replace("selem=selem", "footprint=selem")

    helper_anchor = "import scipy.ndimage\n"
    helper = r'''

def _run_partition_cli(img_path, partnum, window_x, window_y, step):
    """Invoke the unchanged official partition optimizer with this interpreter."""
    optimizer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalize_partition.py")
    command = [
        sys.executable, optimizer,
        "--grp", str(group_name),
        "--lm", str(lmbda),
        "--beta_bnd_l", str(lower_bnd_beta),
        "--beta_bnd_u", str(upper_bnd_beta),
        "--alpha_bnd_l", str(lower_bnd_alpha),
        "--alpha_bnd_u", str(upper_bnd_alpha),
        "--cnst", str(cnst),
        "--fact", str(fact),
        "--beta_val_l", str(lower_val_beta),
        "--beta_val_u", str(upper_val_beta),
        "--alpha_val_l", str(lower_val_alpha),
        "--alpha_val_u", str(upper_val_alpha),
        str(img_path), str(int(partnum)), str(int(window_x)), str(int(window_y)),
        str(int(num_partitions_x)), str(int(num_partitions_y)), str(int(num_parallel)),
        str(int(step)), str(output_path),
    ]
    subprocess.run(command, check=True)
'''
    text = _replace_exact(text, helper_anchor, helper_anchor + "import subprocess\n" + helper)

    substitutions = {
        "size = np.prod(shape)": "size = int(np.prod(shape))",
        "volume_parts = np.ceil(float(volume_size)/ (100*max_shared_array_size))":
            "volume_parts = int(np.ceil(float(volume_size) / (100 * max_shared_array_size)))",
        "for z_division in np.arange(volume_parts):": "for z_division in range(volume_parts):",
        "r=r_org/num_partitions_x": "r = r_org // num_partitions_x",
        "c=c_org/num_partitions_y": "c = c_org // num_partitions_y",
        "alpha=np.zeros((r_org/window_size_x,c_org/window_size_y,s))":
            "alpha = np.zeros((r_org // window_size_x, c_org // window_size_y, s))",
        "beta=np.zeros((r_org/window_size_x,c_org/window_size_y,s))":
            "beta = np.zeros((r_org // window_size_x, c_org // window_size_y, s))",
        "param[len(param)/2:]": "param[len(param) // 2:]",
        "param[0:len(param)/2]": "param[0:len(param) // 2]",
        "(r/window_size_x,c/window_size_y,s)": "(r // window_size_x, c // window_size_y, s)",
        "partnum_x*r/window_size_x": "partnum_x * r // window_size_x",
        "(partnum_x+1)*r/window_size_x": "(partnum_x + 1) * r // window_size_x",
        "partnum_y*c/window_size_y": "partnum_y * c // window_size_y",
        "(partnum_y+1)*c/window_size_y": "(partnum_y + 1) * c // window_size_y",
        "window_size_x=window_size_x/2": "window_size_x //= 2",
        "window_size_y=window_size_y/2": "window_size_y //= 2",
    }
    for old, new in substitutions.items():
        if old not in text:
            raise EmisacCompatibilityError(f"Missing expected integer-division expression: {old}")
        text = text.replace(old, new)

    # The official local worker function relies on Linux ``fork`` semantics.
    # Colab uses Linux and therefore keeps the original process path.  Windows
    # has no ``fork`` and cannot pickle a nested function, so threads provide a
    # portability-only fallback for the local smoke test.
    text = _replace_exact(
        text,
        "p = multiprocessing.Process(target = thread_proc)",
        "worker_type = threading.Thread if os.name == 'nt' else multiprocessing.Process\n                p = worker_type(target=thread_proc)",
    )

    downsample_slice = "img=img_down[((partnum%num_partitions_x)*r_down/num_partitions_x):((partnum%num_partitions_x)*r_down/num_partitions_x+r_down/num_partitions_x),(np.floor(partnum/num_partitions_x)*c_down/num_partitions_y):(np.floor(partnum/num_partitions_x)*c_down/num_partitions_y+c_down/num_partitions_y),:]"
    downsample_slice_new = "img = img_down[((partnum % num_partitions_x) * r_down // num_partitions_x):((partnum % num_partitions_x + 1) * r_down // num_partitions_x), ((partnum // num_partitions_x) * c_down // num_partitions_y):((partnum // num_partitions_x + 1) * c_down // num_partitions_y), :]"
    text = _replace_exact(text, downsample_slice, downsample_slice_new)

    full_slice = "img=img_org[((partnum%num_partitions_x)*r_org/num_partitions_x):((partnum%num_partitions_x)*r_org/num_partitions_x+r_org/num_partitions_x),(np.floor(partnum/num_partitions_x)*c_org/num_partitions_y):(np.floor(partnum/num_partitions_x)*c_org/num_partitions_y+c_org/num_partitions_y),:]"
    full_slice_new = "img = img_org[((partnum % num_partitions_x) * r_org // num_partitions_x):((partnum % num_partitions_x + 1) * r_org // num_partitions_x), ((partnum // num_partitions_x) * c_org // num_partitions_y):((partnum // num_partitions_x + 1) * c_org // num_partitions_y), :]"
    text = _replace_exact(text, full_slice, full_slice_new)

    downsample_call = 'os.system("./normalize_partition.py --grp %s --lm %f --beta_bnd_l %f --beta_bnd_u %f --alpha_bnd_l %f --alpha_bnd_u %f --cnst %f --fact %f --beta_val_l %f --beta_val_u %f --alpha_val_l %f --alpha_val_u %f %s %d %d %d %d %d %d %d %s"%(group_name,lmbda,lower_bnd_beta,upper_bnd_beta,lower_bnd_alpha,upper_bnd_alpha,cnst,fact,lower_val_beta,upper_val_beta,lower_val_alpha,upper_val_alpha,img_path,partnum,window_size_x/step_x,window_size_y/step_y,num_partitions_x,num_partitions_y,num_parallel,step,output_path))'
    text = _replace_exact(
        text,
        downsample_call,
        "_run_partition_cli(img_path, partnum, window_size_x // step_x, window_size_y // step_y, step)",
    )
    full_call = 'os.system("./normalize_partition.py --grp %s --lm %f --beta_bnd_l %f --beta_bnd_u %f --alpha_bnd_l %f --alpha_bnd_u %f --cnst %f --fact %f --beta_val_l %f --beta_val_u %f --alpha_val_l %f --alpha_val_u %f %s %d %d %d %d %d %d %d %s"%(group_name,lmbda,lower_bnd_beta,upper_bnd_beta,lower_bnd_alpha,upper_bnd_alpha,cnst,fact,lower_val_beta,upper_val_beta,lower_val_alpha,upper_val_alpha,img_path,partnum,window_size_x,window_size_y,num_partitions_x,num_partitions_y,num_parallel,step,output_path))'
    text = _replace_exact(
        text,
        full_call,
        "_run_partition_cli(img_path, partnum, window_size_x, window_size_y, step)",
    )
    return text


def _patch_partition(text: str) -> str:
    text = text.expandtabs(4)
    text = _replace_exact(text, "import skimage.filter.rank as rank", "from skimage.filters import rank")
    text = _replace_exact(
        text,
        "from skimage.morphology import rectangle",
        """try:\n    from skimage.morphology import rectangle\nexcept ImportError:\n    from skimage.morphology import footprint_rectangle\n\n    def rectangle(nrows, ncols):\n        return footprint_rectangle((nrows, ncols))""",
    )
    text = _replace_exact(text, "img=d[group_name].value", "img = d[group_name][()]")
    text = _replace_exact(text, "x0=g[group_name].value", "x0 = g[group_name][()]")
    substitutions = {
        "A.shape[0]/ block[0]": "A.shape[0] // block[0]",
        "A.shape[1]/ block[1]": "A.shape[1] // block[1]",
        "partition_num_block_r=(r/window_size_x)": "partition_num_block_r = r // window_size_x",
        "partition_num_block_c=(c/window_size_y)": "partition_num_block_c = c // window_size_y",
        "cnt*r*c/partition_num_rc": "cnt * r * c // partition_num_rc",
    }
    for old, new in substitutions.items():
        if old not in text:
            raise EmisacCompatibilityError(f"Missing expected partition expression: {old}")
        text = text.replace(old, new)
    return text


def patch_official_source(source_dir: Path) -> dict:
    """Patch the pinned checkout in-place and write an auditable manifest."""
    source_dir = source_dir.resolve()
    actual_commit = _git_commit(source_dir)
    if actual_commit != OFFICIAL_COMMIT:
        raise EmisacCompatibilityError(
            f"Refusing to patch non-pinned checkout: expected {OFFICIAL_COMMIT}, found {actual_commit}"
        )
    marker = source_dir / PATCH_MANIFEST
    if marker.exists():
        return verify_prepared_source(source_dir)
    _require_clean_checkout(source_dir)

    main_file = source_dir / "EM_normalization.py"
    partition_file = source_dir / "normalize_partition.py"
    required = [main_file, partition_file, source_dir / "license.txt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EmisacCompatibilityError(f"Official checkout is incomplete: {missing}")

    before = {path.name: _sha256(path) for path in (main_file, partition_file)}
    _python3_refactor([main_file, partition_file])
    main_file.write_text(_patch_main(main_file.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    partition_file.write_text(
        _patch_partition(partition_file.read_text(encoding="utf-8")),
        encoding="utf-8",
        newline="\n",
    )
    _run([sys.executable, "-m", "py_compile", main_file, partition_file])
    after = {path.name: _sha256(path) for path in (main_file, partition_file)}
    manifest = {
        "schema_version": 1,
        "compatibility_version": COMPATIBILITY_VERSION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "patch_scope": [
            "Python 2 print statements converted to Python 3 syntax",
            "Python 2 integer-division semantics retained for shapes and indices",
            "Pillow, h5py, NumPy, and scikit-image API names updated",
            "partition subprocess launched with the active Python interpreter",
            "Windows-only thread fallback for the official nested smoothing worker",
        ],
        "algorithm_changes": "none",
        "sha256_before": before,
        "sha256_after": after,
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify_prepared_source(source_dir)


def verify_prepared_source(source_dir: Path) -> dict:
    """Verify the pinned commit and the hashes of the compatibility-patched files."""
    source_dir = source_dir.resolve()
    actual_commit = _git_commit(source_dir)
    if actual_commit != OFFICIAL_COMMIT:
        raise EmisacCompatibilityError(
            f"Commit verification failed: expected {OFFICIAL_COMMIT}, found {actual_commit}"
        )
    marker = source_dir / PATCH_MANIFEST
    if not marker.is_file():
        raise EmisacCompatibilityError(f"Missing compatibility manifest: {marker}")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if manifest.get("compatibility_version") != COMPATIBILITY_VERSION:
        raise EmisacCompatibilityError("Compatibility version mismatch; prepare a fresh work directory.")
    if manifest.get("official_commit") != OFFICIAL_COMMIT:
        raise EmisacCompatibilityError("Manifest official commit does not match the launcher pin.")
    expected_files = {"EM_normalization.py", "normalize_partition.py"}
    if (set(manifest.get("sha256_before", {})) != expected_files
            or set(manifest.get("sha256_after", {})) != expected_files):
        raise EmisacCompatibilityError("Manifest must record both official script hashes.")
    for filename, expected in manifest.get("sha256_after", {}).items():
        actual = _sha256(source_dir / filename)
        if actual != expected:
            raise EmisacCompatibilityError(
                f"Prepared source hash mismatch for {filename}: expected {expected}, found {actual}"
            )
    print(f"[EMISAC official] verified commit {actual_commit}")
    return manifest


def prepare_source(work_dir: Path) -> Path:
    source_dir = work_dir.resolve() / "official_emisac_source"
    if not source_dir.exists():
        clone_official_source(source_dir)
    if (source_dir / PATCH_MANIFEST).exists():
        verify_prepared_source(source_dir)
    else:
        patch_official_source(source_dir)
    return source_dir


def _input_images(input_dir: Path) -> list[Path]:
    images = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    if len(images) < 2:
        raise EmisacCompatibilityError(f"EMISAC needs at least two slices; found {len(images)} in {input_dir}")
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise EmisacCompatibilityError("Input filenames have duplicate stems; output names would collide.")
    return images


def _build_tiff(images: Sequence[Path], output_tiff: Path) -> tuple[int, int]:
    from PIL import Image

    pages = []
    size = None
    for path in images:
        with Image.open(path) as image:
            page = image.convert("L")
            if size is None:
                size = page.size
            elif page.size != size:
                raise EmisacCompatibilityError(
                    f"All input slices must share one shape; {path.name} is {page.size}, expected {size}"
                )
            pages.append(page.copy())
    assert size is not None
    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    # Keep the interchange TIFF uncompressed.  This mirrors the simple TIFFs
    # expected by the 2014 PIL loader and avoids codec-specific behavior.
    pages[0].save(output_tiff, save_all=True, append_images=pages[1:], compression="raw")
    return size[1], size[0]


def _validate_geometry(
    height: int,
    width: int,
    *,
    step_x: int,
    step_y: int,
    window_x: int,
    window_y: int,
    partitions_x: int,
    partitions_y: int,
    workers: int,
) -> None:
    values = [step_x, step_y, window_x, window_y, partitions_x, partitions_y, workers]
    if any(value <= 0 for value in values):
        raise EmisacCompatibilityError("All sampling, window, partition, and worker values must be positive.")
    if workers > partitions_x * partitions_y:
        raise EmisacCompatibilityError("workers cannot exceed partitions_x * partitions_y")
    checks = [
        (height % step_x == 0, "height must be divisible by step_x"),
        (width % step_y == 0, "width must be divisible by step_y"),
        (window_x % step_x == 0, "window_x must be divisible by step_x"),
        (window_y % step_y == 0, "window_y must be divisible by step_y"),
        (height % partitions_x == 0, "height must be divisible by partitions_x"),
        (width % partitions_y == 0, "width must be divisible by partitions_y"),
        ((height // partitions_x) % window_x == 0, "each x partition must be divisible by window_x"),
        ((width // partitions_y) % window_y == 0, "each y partition must be divisible by window_y"),
        (((height // step_x) // partitions_x) % (window_x // step_x) == 0,
         "each downsampled x partition must be divisible by its window"),
        (((width // step_y) // partitions_y) % (window_y // step_y) == 0,
         "each downsampled y partition must be divisible by its window"),
    ]
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise EmisacCompatibilityError("Invalid official EMISAC geometry: " + "; ".join(failed))


def _export_h5(h5_path: Path, images: Sequence[Path], output_dir: Path) -> list[Path]:
    import h5py
    import numpy as np
    from PIL import Image

    if not h5_path.is_file():
        raise EmisacCompatibilityError(f"Official EMISAC did not create {h5_path}")
    with h5py.File(h5_path, "r") as handle:
        if "stack" not in handle:
            raise EmisacCompatibilityError(f"Official output has no 'stack' dataset: {h5_path}")
        volume = np.asarray(handle["stack"][()])
    if volume.ndim != 3 or volume.shape[2] != len(images):
        raise EmisacCompatibilityError(
            f"Unexpected official output shape {volume.shape}; expected (H, W, {len(images)})"
        )
    volume = np.clip(np.rint(volume), 0, 255).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, source_path in enumerate(images):
        target = output_dir / f"{source_path.stem}.png"
        Image.fromarray(volume[:, :, index], mode="L").save(target)
        outputs.append(target)
    return outputs


def run_official(args: argparse.Namespace) -> dict:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve()
    if not input_dir.is_dir():
        raise EmisacCompatibilityError(f"Input directory does not exist: {input_dir}")
    images = _input_images(input_dir)
    source_dir = prepare_source(work_dir)
    compatibility = verify_prepared_source(source_dir)

    run_dir = work_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_tiff = run_dir / "input_stack.tif"
    official_output = run_dir / "official_output"
    official_output.mkdir(parents=True, exist_ok=True)
    height, width = _build_tiff(images, input_tiff)
    _validate_geometry(
        height,
        width,
        step_x=args.step_x,
        step_y=args.step_y,
        window_x=args.window_x,
        window_y=args.window_y,
        partitions_x=args.partitions_x,
        partitions_y=args.partitions_y,
        workers=args.workers,
    )

    command = [
        sys.executable,
        source_dir / "EM_normalization.py",
        "--up_fg", "0",
        "--up_z", "1",
        "--fact", str(args.fact),
        "--lm", str(args.lm),
        input_tiff,
        str(args.step_x),
        str(args.step_y),
        str(args.window_x),
        str(args.window_y),
        str(args.partitions_x),
        str(args.partitions_y),
        str(args.workers),
        official_output,
    ]
    started = time.time()
    _run(command, cwd=source_dir)
    elapsed = time.time() - started
    outputs = _export_h5(official_output / "normalized_data_final.h5", images, output_dir)
    metadata = {
        "schema_version": 1,
        "method": "EMISAC",
        "implementation": "authors' official source with compatibility-only patch",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "compatibility_version": COMPATIBILITY_VERSION,
        "compatibility_manifest_sha256": _sha256(source_dir / PATCH_MANIFEST),
        "official_source_sha256_before": compatibility["sha256_before"],
        "patched_source_sha256_after": compatibility["sha256_after"],
        "executed_command": [str(part) for part in command],
        "algorithm_changes": "none",
        "parameters": {
            "up_fg": 0,
            "up_z": 1,
            "fact": args.fact,
            "lm": args.lm,
            "step_x": args.step_x,
            "step_y": args.step_y,
            "window_x": args.window_x,
            "window_y": args.window_y,
            "partitions_x": args.partitions_x,
            "partitions_y": args.partitions_y,
            "workers": args.workers,
        },
        "input": {
            "directory": str(input_dir),
            "slice_count": len(images),
            "height": height,
            "width": width,
            "filenames": [path.name for path in images],
        },
        "output_filenames": [path.name for path in outputs],
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[EMISAC official] wrote {len(outputs)} slices to {output_dir}")
    return metadata


def _smoke_input(input_dir: Path, count: int = 4, size: int = 64) -> None:
    import numpy as np
    from PIL import Image

    input_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    for index in range(count):
        base = 50 + 0.7 * xx + 0.4 * yy + 12 * np.sin(xx / 7.0)
        stripe_gain = 0.72 if index == 1 else (1.18 if index == 2 else 1.0)
        image = np.clip(base * stripe_gain + index, 0, 255).astype(np.uint8)
        Image.fromarray(image, mode="L").save(input_dir / f"slice_{index:05d}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="clone, pin, patch, and verify official source")
    prepare.add_argument("--work-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify an already prepared official source")
    verify.add_argument("--work-dir", type=Path, required=True)

    def add_run_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--input-dir", type=Path, required=True)
        command_parser.add_argument("--output-dir", type=Path, required=True)
        command_parser.add_argument("--work-dir", type=Path, required=True)
        command_parser.add_argument("--workers", type=int, default=2)
        command_parser.add_argument("--step-x", type=int, default=4)
        command_parser.add_argument("--step-y", type=int, default=4)
        command_parser.add_argument("--window-x", type=int, default=32)
        command_parser.add_argument("--window-y", type=int, default=32)
        command_parser.add_argument("--partitions-x", type=int, default=4)
        command_parser.add_argument("--partitions-y", type=int, default=4)
        command_parser.add_argument("--fact", type=float, default=1e10)
        command_parser.add_argument("--lm", type=float, default=100.0)

    run_parser = subparsers.add_parser("run", help="run official EMISAC and export aligned PNG slices")
    add_run_arguments(run_parser)

    smoke = subparsers.add_parser("smoke-test", help="run the complete official pipeline on a tiny volume")
    smoke.add_argument("--work-dir", type=Path, required=True)
    smoke.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            source = prepare_source(args.work_dir)
            print(json.dumps(verify_prepared_source(source), indent=2))
        elif args.command == "verify":
            source = args.work_dir.resolve() / "official_emisac_source"
            print(json.dumps(verify_prepared_source(source), indent=2))
        elif args.command == "run":
            run_official(args)
        elif args.command == "smoke-test":
            smoke_root = args.work_dir.resolve() / "smoke_test"
            input_dir = smoke_root / "input"
            output_dir = smoke_root / "output"
            _smoke_input(input_dir)
            run_args = argparse.Namespace(
                input_dir=input_dir,
                output_dir=output_dir,
                work_dir=args.work_dir,
                workers=args.workers,
                step_x=4,
                step_y=4,
                window_x=32,
                window_y=32,
                partitions_x=2,
                partitions_y=2,
                fact=1e10,
                lm=100.0,
            )
            run_official(run_args)
            print(f"[EMISAC official] smoke test passed: {output_dir}")
        return 0
    except (EmisacCompatibilityError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[EMISAC official] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
