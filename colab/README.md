# Colab inference and comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Pipeline_Demo.ipynb)

Open `vEMDestripe_Bumbarger_Pipeline_Demo.ipynb` to inspect Bumbarger13
sections 777–808 (32 × 1024 × 1024) and compare vEMDestripe, EMISAC,
FECNet and SSCOR.

The workflow is network inference → inference postprocessing → saved outputs
→ evaluation. The notebook provides XY, XZ and YZ views, section profiles,
source code, a comparison table and a downloadable result ZIP.

## Run

1. Open the notebook, connect a CPU or GPU runtime, and select **Runtime → Run all**.
2. Keep `VEM_RUN_MODE = 'published_full_volume'` to use verified saved network
   outputs. This default works on a CPU runtime.
3. Wait for the versioned asset download, SHA256 checks and inference postprocessing.
4. Inspect **Final pipeline outputs** and the comparison table. The
   **Network-only diagnostic** view shows the vEMDestripe intermediate.
5. Download final images, processing receipts and `metrics_final_outputs.csv`.

Optional `VEM_RUN_MODE = 'live_demo'` runs the released checkpoint on the
32 supplied sections with a GPU, then runs inference postprocessing.
Use a fresh runtime when switching modes.

## Processing provenance

This notebook retains the historical **16×16-block checkpoint** with SHA256
`9a01cdef1f09716fe87369b057633f9e1278e86e77cbc70eb66b75798709c44a`.
The default mode uses the complete 2651-section network output as its source.
Live mode uses all 32 newly inferred sections. The result ZIP includes the
saved outputs, processing parameters and hashes in `CALIBRATION.json`,
network intermediates and the versioned source records.

## Comparison outputs

- EMISAC uses verified official source and a fixed legacy environment. Its
  H5, source/GPL, parameters and logs are in `provenance/emisac_official/`.
  See [reproduction instructions](official_emisac/REPRODUCE.md).
- FECNet uses the delivered grayscale EM adaptation and Bumbarger fine-tuned
  output. SSCOR uses its dataset-specific seed-42 output.
- The table reports NIQE, native-Z continuity diagnostics, contrast and blur
  for the 32-section demo. Cross-plane evaluation and views label their 6× Z
  enlargement. The real dataset provides unpaired reference sections.

The executable source is pinned to **`optimized-inference-v1`**. The notebook
uses `vemdestripe_bumbarger_reference_pipeline_v3.zip` from the immutable
**`demo-reference-v3`** release, with ZIP SHA256
`9aa8bdbf82dbbafb4473df73abdcc2dea6a9853c4c6aa135cb4a8809f61ff669`.
Each bundled file is checked against its manifest.

See [发布与使用.md](发布与使用.md), [THIRD_PARTY_METHODS.md](THIRD_PARTY_METHODS.md)
and [DATA_LICENSE.md](DATA_LICENSE.md) for release, implementation and data details.
