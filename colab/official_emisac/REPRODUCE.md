# Reproduce the original-source EMISAC demo

The public Colab reads the **precomputed output of the unmodified original
Python 2 source**. It does not label the Python 3 port as an equivalent original
run. Raw H5 data, original source (with its GPL-3.0 license), PNGs, input TIFF,
parameters, environment details, and logs are included in the release asset.

The fixed source is upstream commit
`7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5` at
<https://github.com/azadis/EMISAC>. The original child-script shebang uses
`/usr/bin/python`; the container therefore supplies the old interpreter at
that path without changing either source file. Merely starting the parent
script with a different Python executable is insufficient.

## Prerequisites

Use Docker with Linux x86-64 containers and at least 8 GB available RAM. The
host launcher uses Python 3 with `numpy`, `h5py`, and `Pillow`. It is intended
for a workstation/server, not an untested nested-Docker setup in hosted Colab.
No GPU is required. Downloads/builds require network access; the actual
algorithm is run offline.

From the repository root, after extracting the release archive:

```bash
python colab/official_emisac/reproduce.py \
  --asset-root /absolute/path/vemdestripe_bumbarger_official_demo_v2 \
  --output-dir /absolute/path/new_emisac_reproduction \
  --build
```

`--build` creates `vemdestripe-emisac-original:demo-v2` from this directory's
Dockerfile; no unpublished parent image is required. It downloads the fixed
Anaconda 2.1.0 installer and joblib/scikit-image archives with SHA256 checks.
Read the original distributors' terms before downloading these dependencies.
The source and input are mounted read-only. Existing output directories are
never overwritten. Use `--image IMAGE` to run an already built matching image.

The output directory contains a log, the raw output under `output/`, PNGs
under `png/`, and `comparison.json` reporting whether the new H5 equals the
bundled reference. A completed run is **not** assumed numerically identical:
always read the reported differences. Rebuilding old native code on a
different base/hardware can change numerical results.

## Scope and interpretation

- Exactly local Z `[777,809)` (32 slices) is processed, not the full dataset.
- Sampling 4×4, initial windows 32×32, partitions 4×4, 2 workers, `lm=100`,
  `fact=1e10`, `up_fg=0`, inactive `up_z=1`, `cns_fg=0`.
- These are explicit matched settings, **not all upstream defaults**; default
  Z upsampling is deliberately disabled.
- PNG conversion is only `clip(rint(H5),0,255).astype(uint8)`; no additional
  brightness/histogram matching is applied to the raw reference.
- This documents a runnable legacy reference environment, not recovery of
  every detail of the authors' historical computer.
- Old dependencies are isolated for this reproduction exercise. Do not expose
  the container as a service or use this environment for unrelated workloads.

The previously demonstrated Python 3 port did not reproduce these real-data
outputs. Its live run is not used in the new official-reference comparison.
