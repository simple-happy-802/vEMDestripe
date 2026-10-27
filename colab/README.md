# Google Colab comparison demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Official_Demo.ipynb)

`vEMDestripe_Bumbarger_Official_Demo.ipynb` compares vEMDestripe, EMISAC,
FECNet, and SSCOR on 32 Bumbarger13 sections, local Z `[777,809)`
(777–808 inclusive), at 1024×1024 pixels.

## Run the demo

1. Open the notebook above and connect a Colab runtime.
2. Use a GPU runtime for live vEMDestripe inference, or a CPU runtime to view
   the explicitly labelled packaged vEMDestripe output.
3. Select **Runtime → Run all**. The notebook downloads and verifies the
   release bundle before loading images.
4. Inspect XY/XZ/YZ views, Z profiles, source code, and provenance. The default
   view is **Raw outputs**, without an extra photometric transform.
5. Download the result archive from the export cell. Raw and globally matched
   metrics are labelled separately.

## Which results are computed live?

| Method | Default result used by this notebook |
| --- | --- |
| vEMDestripe | Live inference with the released checkpoint when enabled and a GPU is available; otherwise the clearly labelled frozen output. |
| EMISAC | Precomputed output of the **unmodified official Python 2 source** in a recorded legacy environment. This is not a live Colab run. |
| FECNet | Frozen output from the grayscale EM adaptation and Bumbarger-specific fine-tuning used for the comparison. |
| SSCOR | Frozen output from the dataset-specific self-supervised comparison run. |

The notebook recomputes the displayed metrics from the selected images. It
does not retrain the comparison methods or imply that all methods are rerun
inside Colab. The source viewer shows the corresponding source entry points;
the provenance record identifies how the packaged outputs were generated.

## Official EMISAC reference

The EMISAC output was generated from official commit
`7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5`, without editing the two upstream
scripts. The bundle includes the original floating-point H5, the input TIFF,
source and license, run log, command, environment versions, and SHA256
manifest. Display PNGs use `clip(rint(H5), 0, 255).astype(uint8)`; they receive
no additional per-section brightness adjustment.

This run processes the 32-section input itself, with step 4×4, initial window
32×32, partitions 4×4, `lm=100`, `fact=1e10`, two workers, and **Z upsampling
disabled** (`up_fg=0`). These are explicit experiment parameters, not a claim
to use every upstream default. The recorded legacy environment is not claimed
to reconstruct every detail of the authors' historical machine.

To rerun the original algorithm outside hosted Colab, see
[official_emisac/REPRODUCE.md](official_emisac/REPRODUCE.md). The earlier
`vEMDestripe_Bumbarger_Demo.ipynb` and `run_official_emisac.py` remain as
historical Python 3 compatibility experiments; the compatibility port did not
pass real-data numerical-equivalence testing and is not the official reference
in this new demo.

## Display and metric scope

Raw outputs are the default. **Global affine matched** is an optional,
explicitly labelled view using one scale/bias per entire method volume, never
independent stretching of individual slices. Both protocols have separately
labelled metric results; a matched image is not an unchanged algorithm output.

This real volume has no slice-wise clean ground truth, so PSNR and SSIM are
not reported. Demo metrics must not replace the paper's 200-section results.
`ASID_MAD` measures the median absolute deviation of adjacent whole-section
mean changes, not the benchmark's pixelwise ASID percentage. `local_ASID`
uses local jumps divided by local IQR. Cross-plane NIQE uses a labelled 6× Z
enlargement; native-Z continuity is computed without interpolation. Contrast
and blur are diagnostics, not biological ground truth.

The displayed section range is shared, but processing context can differ
between frozen comparison runs; consult the bundle provenance rather than
assuming that each method was given only these 32 sections.

## Release configuration

The notebook is configured for `simple-happy-802/vEMDestripe`, branch `main`,
release tag `demo-official-v2`, and asset
`vemdestripe_bumbarger_official_demo_v2.zip`. Code belongs in the repository;
the image/H5 bundle belongs in the GitHub Release, not ordinary Git history.
Its archive SHA256 is pinned by the notebook, and `manifest.json` records the
individual bundled files.

See [发布与使用.md](发布与使用.md) for publishing and reader instructions,
[THIRD_PARTY_METHODS.md](THIRD_PARTY_METHODS.md) for baseline provenance, and
[DATA_LICENSE.md](DATA_LICENSE.md) for data attribution. A local check is not
a claim that the hosted Colab runtime has been tested; after publishing, run
the public notebook from a fresh Colab session before citing its link.
