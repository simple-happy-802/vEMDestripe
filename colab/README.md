# Complete-pipeline Colab comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Pipeline_Demo.ipynb)

The recommended notebook is `vEMDestripe_Bumbarger_Pipeline_Demo.ipynb`.
It compares local Bumbarger13 sections 777–808 (32 × 1024 × 1024).

## What is evaluated

The vEMDestripe pipeline is:

`network prediction → whole-volume reference mean/std calibration → save final images → evaluate saved files`

Only vEMDestripe includes this output-calibration stage. Original, EMISAC,
FECNet and SSCOR retain their delivered pixels. The evaluator never fits or
applies a brightness/contrast transformation, including to vEMDestripe.

The target is the pooled pixels of the metadata-selected **original reference
snapshots**, not the fused guidance image, its 16×16 block-averaged version,
or the test input volume. Each output section uses the same map:

```python
scale = reference_std / output_std
bias = reference_mean - scale * output_mean
lut = np.rint(np.clip(np.arange(256, dtype=np.float32) * scale + bias, 0, 255)).astype(np.uint8)
final_image = lut[saved_network_output]
```

The float32 LUT and rounding order match the recorded paper export pipeline.
No independent per-slice fitting is performed.

## Run

1. Open the notebook, connect a CPU or GPU runtime, and select **Runtime → Run all**.
2. Keep `VEM_RUN_MODE = 'published_full_volume'` to reproduce the recorded
   complete-volume calibration. No GPU is required for this default.
3. Wait for the versioned asset download and SHA256 checks. The notebook
   materializes final PNGs, a TIFF and `CALIBRATION.json` before scoring.
4. Inspect XY/XZ/YZ, brightness profiles, the source viewer and the final table.
   **Final pipeline outputs** is the primary view. **Network-only diagnostic**
   changes only the vEMDestripe view, not the evaluation input.
5. Download the result ZIP containing final outputs, the separate network
   intermediate, calibration provenance and `metrics_final_outputs.csv`.

Optional `VEM_RUN_MODE = 'live_demo'` requires a GPU and runs the released
checkpoint on all 32 supplied sections, followed by the same calibration
algorithm on this new 32-section output volume. This mode is explicitly a
different source-statistics context, not the paper's 2651-section calibration.
Use a fresh runtime when switching modes.

## Recorded full-volume calibration

The default reproduces the selected **16×16-block checkpoint**:

| Quantity | Value |
| --- | ---: |
| Complete network-output sections | 2651 |
| Original reference snapshots | 11 |
| Source mean | 146.69773852119172 |
| Source population std | 56.64240473977118 |
| Reference pooled mean | 90.97551380504262 |
| Reference pooled population std | 35.92554044163754 |
| Scale | 0.6342516813452413 |
| Bias | -2.0677735015677996 |

All 2651 saved outputs were reread for source histograms and file hashes.
The 32 displayed network files and 11 reference files were checked against
their original inventories. Calibration precedes the display crop; the
32-section window is never used to refit the published map. Evidence is in
the release's `calibration/` directory.

## Baselines and limits

- EMISAC is the verified unmodified official Python-2 source in a fixed legacy
  environment, precomputed on the 32-section input. No additional calibration
  is applied. Original H5, source/GPL, explicit parameters and logs remain in
  `provenance/emisac_official/`. See [official_emisac/REPRODUCE.md](official_emisac/REPRODUCE.md).
- FECNet is the delivered grayscale EM adaptation/Bumbarger fine-tuned output;
  SSCOR is its dataset-specific seed-42 comparison output. They are not live
  training runs in this notebook, and do not receive vEMDestripe's calibration.
- This real dataset has no paired clean target. No PSNR/SSIM is reported.
  NIQE, native-Z ASID_MAD/local-ASID/Z-HF and contrast/blur are diagnostics,
  not biological ground truth. Cross-plane NIQE/views explicitly enlarge Z 6×.
- The 32-section demonstration is not the paper's full benchmark. Shared display
  sections do not imply identical original processing extents for all methods.

Code is pinned to `demo-reference-v3`; the Release asset is
`vemdestripe_bumbarger_reference_pipeline_v3.zip`, with ZIP SHA256 pinned in
the notebook and every file checked against its manifest. Previous v2 and
Python-3-port notebooks remain historical, not the recommended complete pipeline.

See [发布与使用.md](发布与使用.md), [THIRD_PARTY_METHODS.md](THIRD_PARTY_METHODS.md)
and [DATA_LICENSE.md](DATA_LICENSE.md). Local execution does not establish that
the Google-hosted GPU runtime has been tested.
