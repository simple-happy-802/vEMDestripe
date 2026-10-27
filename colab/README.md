# Google Colab comparison demo

`vEMDestripe_Bumbarger_Demo.ipynb` is the public, click-through comparison of
vEMDestripe, EMISAC, FECNet, and SSCOR on Bumbarger13.

## What the notebook does

- downloads a fixed 32-section subset inside the paper's audited Bumbarger13
  evaluation window;
- clones the authors' official EMISAC repository at a recorded commit and runs
  its original optimization pipeline on the demo subset;
- runs the released vEMDestripe checkpoint when a GPU is available;
- loads immutable FECNet and SSCOR outputs produced by the paper
  configurations;
- provides interactive XY, XZ, and YZ views and native-Z brightness curves;
- recomputes demo-window ASID, local-ASID, Z-HF, NIQE XY/XZ/YZ, contrast, and
  blur diagnostics;
- keeps an optional, hidden historical manuscript table for provenance only; and
- links to and displays the source entry points for every method.

The Bumbarger volume has no slice-wise clean ground truth. PSNR and SSIM are
therefore intentionally absent.

EMISAC is not replaced by the local GPU reimplementation in this notebook.
The notebook verifies the upstream commit before applying a small, displayed
Python-3 compatibility patch to a temporary working tree. The patch updates
obsolete imports, syntax, and integer-division semantics only; it does not
change the objective or numerical parameters. The source commit, pre-/post-patch
hashes, compatibility-manifest hash, and exact command are recorded with the
result.

## Metric scope

The live table uses local Z `[777,809)` and one global scale/bias match per
32-section volume. It scores 8 XY sections and 6 interior sections in each
orthogonal orientation. Cross-plane NIQE uses a labelled 6x Z enlargement;
native-Z continuity is never computed after interpolation. `ASID_MAD` is the
median absolute deviation of adjacent whole-section mean changes (not the
benchmark's pixelwise ASID percentage). `local_ASID` uses median jumps divided
by local IQR. Contrast and blur are diagnostics, not biological ground truth.
Raw and globally matched image displays are both available.

## Before publishing

The notebook and launch badges are configured for
`https://github.com/simple-happy-802/vEMDestripe`, branch `main`. These local
configuration changes do not upload files or publish a Release. To publish:

1. Upload the contents of the local `vEMDestripe` folder to the root of
   `https://github.com/simple-happy-802/vEMDestripe` on branch `main`.
2. Create a GitHub Release tagged `demo-v1`.
3. Upload `vemdestripe_bumbarger_demo_z0777_z0808.zip` as a release asset. The
   archive contains the input, references, FECNet/SSCOR comparison outputs,
   and a vEMDestripe fallback, but no precomputed EMISAC result.
4. Keep the configured owner `simple-happy-802`, repository `vEMDestripe`,
   branch `main`, and Release tag `demo-v1`, unless intentionally relocating.
5. Open the final notebook in Colab and run **Runtime -> Run all** once.

The public URL will be:

```text
https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Demo.ipynb
```

See [THIRD_PARTY_METHODS.md](THIRD_PARTY_METHODS.md) for baseline provenance
and [DATA_LICENSE.md](DATA_LICENSE.md) for the ODC-By attribution.
