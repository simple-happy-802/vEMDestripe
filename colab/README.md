# Colab comparison with one fixed Original target

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Pipeline_Demo.ipynb)

The recommended notebook is `vEMDestripe_Bumbarger_Pipeline_Demo.ipynb`.
It compares local Bumbarger13 sections 777–808 (32 × 1024 × 1024).

## What is evaluated

This edition uses the user-selected **Original `slice_00793.png`** as the
vEMDestripe brightness/contrast target. Its demonstration pipeline is:

`network prediction → one whole-volume mean/std map to Original slice_00793 → save final images → evaluate saved files`

Only vEMDestripe includes this output-calibration stage. Original, EMISAC,
FECNet and SSCOR retain their delivered pixels. The evaluator never fits or
applies a brightness/contrast transformation, including to vEMDestripe.

The target mean and population standard deviation are calculated from every
pixel in **`input/slice_00793.png`**, not the 11-reference pool, a fused or
block-averaged guidance image, or a separate target for each output slice.
Each output section uses the same map:

```python
scale = original_00793_std / output_std
bias = original_00793_mean - scale * output_mean
lut = np.rint(np.clip(np.arange(256, dtype=np.float32) * scale + bias, 0, 255)).astype(np.uint8)
final_image = lut[saved_network_output]
```

The float32 LUT and rounding order retain the audited export implementation.
This is intensity calibration, not spatial registration. No independent
per-slice fitting, image warping or registration of biological structures is
performed. The chosen target is a demonstration setting, **not the earlier
paper reference-pool calibration protocol**. Training and the default
reference-based behavior of `infer.py` are unchanged.

## Run

1. Open the notebook, connect a CPU or GPU runtime, and select **Runtime → Run all**.
2. Keep `VEM_RUN_MODE = 'published_full_volume'` to use the recorded complete
   2651-section network-output statistics with the fixed Original-793 target.
   No GPU is required for this default.
3. Wait for the versioned asset download and SHA256 checks. The notebook
   computes the Original-793 target statistics and materializes new final
   PNGs, a TIFF and `CALIBRATION.json` before scoring.
4. Inspect XY/XZ/YZ, brightness profiles, the source viewer and the final table.
   **Final pipeline outputs** is the primary view. **Network-only diagnostic**
   changes only the vEMDestripe view, not the evaluation input.
5. Download the result ZIP containing final outputs, the separate network
   intermediate, calibration provenance and `metrics_final_outputs.csv`.

Optional `VEM_RUN_MODE = 'live_demo'` requires a GPU and runs the released
checkpoint on all 32 supplied sections, followed by one map from this new
32-section output volume to the **same fixed Original-793 target**. This mode
is explicitly a different source-statistics context, not the 2651-section fit.
Use a fresh runtime when switching modes.

## Calibration source and target

The default uses the selected **16×16-block checkpoint** and preserves its
complete-output source statistics:

| Quantity | Value |
| --- | ---: |
| Complete network-output sections | 2651 |
| Source mean | 146.69773852119172 |
| Source population std | 56.64240473977118 |
| Fixed target | `input/slice_00793.png` |
| Target section count | 1 |

All 2651 saved outputs were reread for source histograms and file hashes.
The 32 displayed network files were checked against their original inventory.
The target's mean/std, file hash, source scope, scale, bias and mapping table
are recorded with the newly saved outputs. The default source statistics are
not refitted on the 32-section display crop.

The immutable v3 bundle still contains its historical 11-reference calibration
record and calibrated PNGs for provenance checks. **Those old final PNGs and
that old target mapping are not the v4 result.** The notebook creates new
Original-793-calibrated files in its new result directory, and the evaluator
reloads those new files.

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

Code is pinned to **`demo-original-target-v4`**. To avoid another large duplicate
archive, this edition reuses `vemdestripe_bumbarger_reference_pipeline_v3.zip`
from the immutable **`demo-reference-v3`** Release. Its pinned ZIP SHA256 is
`9aa8bdbf82dbbafb4473df73abdcc2dea6a9853c4c6aa135cb4a8809f61ff669`, and every
bundled file is checked against its manifest. Historical tags remain unchanged.
The earlier v3 edition retains the reference-pool target; it must not be
confused with this user-selected Original-target demonstration.

See [发布与使用.md](发布与使用.md), [THIRD_PARTY_METHODS.md](THIRD_PARTY_METHODS.md)
and [DATA_LICENSE.md](DATA_LICENSE.md). Local execution does not establish that
the Google-hosted GPU runtime has been tested.
