# Evaluation protocol

`evaluation/evaluate.py` is the packaged publication benchmark derived from the two supplied `evaluate_zstripe_benchmark*.py` programs. It discovers available methods, checks stack alignment, computes metrics, caches expensive measurements, and writes tables, curves, figures, and provenance files.

## Input roots

Paths are controlled by environment variables rather than user-specific absolute paths:

| Variable | Default | Purpose |
|---|---|---|
| `VEMDESTRIPE_TEST_DATA_ROOT` | `test_data` | original real/paired inputs |
| `VEMDESTRIPE_SIM_DATA_ROOT` | `testdata-sim` | separate weak/medium/strong simulations |
| `VEMDESTRIPE_RESULT_ROOT` | `results/vemdestripe` | vEMDestripe outputs |
| `VEMDESTRIPE_GT_ROOT` | `data/ground_truth` | aligned full-reference volumes |
| `AIODEFLICKER_RESULT_ROOT` | `results/all-in-one-deflicker` | optional baseline |
| `EMISAC_RESULT_ROOT` | external sibling path | optional baseline |
| `FECNET_RESULT_ROOT` | external sibling path | optional baseline |

Unavailable optional baselines are skipped. The evaluator never fabricates missing results.

## Metrics

For paired aligned simulations, the benchmark can report PSNR, SSIM, and FRC-based frequency fidelity together with photometric and axial statistics. For real volumes without clean targets, it reports axial continuity/stripe indices, Z-profile variation, histogram divergence, no-reference image-quality measurements, and XZ/YZ appearance summaries. Expensive NIQE measurements use a cached pristine model selected independently from held-out evaluation sections.

The evaluator resolves the same input layouts as batch inference. For full-reference datasets it first checks `<test-root>/<dataset>/groundtruth` and `<test-root>/<dataset>/test/clean`, then falls back to the named TIFF mappings under `VEMDESTRIPE_GT_ROOT`.

Six explicitly configured anisotropic dataset families use 6× Z interpolation only for XZ/YZ appearance measures and rendered orthogonal views. Native-Z continuity statistics are never computed on the interpolated stack.

## Alignment and reference integrity

- Compare the same Z indices for every method.
- Do not silently crop or reorder a method to improve its result.
- Record the continuous evaluation window selected by `max-hf-mad`.
- Use ground truth only for full-reference metrics and a declared paired-reference experiment.
- Do not use labels to fit display matching or to select a favorable real-data window.
- Report raw vEMDestripe output as the primary result; any global intensity matching is a separately named diagnostic.

## Command

```bash
export VEMDESTRIPE_TEST_DATA_ROOT=/path/to/test_data
export VEMDESTRIPE_RESULT_ROOT=/path/to/results/vemdestripe
export VEMDESTRIPE_GT_ROOT=/path/to/ground_truth
python -m evaluation.evaluate \
  --output-dir evaluation_output/vemdestripe \
  --gpu-batch-size 8 \
  --evaluation-window-mode max-hf-mad \
  --display-mode raw
```

Restrict datasets with a comma-separated list:

```bash
python -m evaluation.evaluate --output-dir evaluation_output/subset \
  --datasets EPFL_center1024,jrc_mus-kidney-medium
```

Optional BRISQUE/PIQE metrics require `requirements-evaluation.txt`. Core evaluation remains usable when those packages are absent.
