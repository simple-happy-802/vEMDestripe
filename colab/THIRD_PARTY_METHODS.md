# Third-party methods in the Bumbarger13 demo

The recommended notebook is `vEMDestripe_Bumbarger_Official_Demo.ipynb`.
It displays source entry points and records which images were generated live
and which are immutable precomputed artifacts. Showing source does not imply
that every method is rerun inside the notebook.

| Method | Public source | License status recorded during packaging | Demo provenance |
| --- | --- | --- | --- |
| EMISAC | <https://github.com/azadis/EMISAC> | GPL-3.0; bundled original scripts retain the upstream license. | Unmodified official commit `7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5`, run in the recorded legacy environment on the 32-section demo input. Its output is precomputed, not a live Colab run. |
| FECNet | <https://github.com/KevinJ-Huang/FECNet> | No license file was visible in the checked public repository; source is linked rather than redistributed. | Frozen output from the grayscale EM adaptation and Bumbarger-specific fine-tuning used for the comparison, not an unmodified default-weight RGB run. |
| SSCOR | <https://github.com/AI-XPM-group/SSCOR> | Apache-2.0 in the checked repository. | Frozen output from dataset-specific self-supervised training, seed 42. |

## EMISAC: original-source reference

The release bundle retains the input TIFF, original floating-point result H5,
unmodified upstream source and license, command, run log, dependency versions,
and checksums under `provenance/emisac_official/`. The PNGs shown in the
notebook are a fixed quantization of this H5, without per-section stretching.
The manifest lets readers verify both the source and result files.

The run uses step 4×4, initial window 32×32, partitions 4×4, `lm=100`,
`fact=1e10`, two workers, and `up_fg=0`. In particular, Z upsampling is
disabled; this is an explicitly specified configuration, not every upstream
default. The fixed legacy dependency environment is a runnable reference,
not a claim to restore the authors' entire historical machine.

See [official_emisac/REPRODUCE.md](official_emisac/REPRODUCE.md) to rerun the
original algorithm. The new notebook does not require changing the hosted
Colab system Python or installing historical packages into its main kernel.

The previous notebook's Python 3 compatibility runner remains available for
historical experiments, but real-data regression testing found nonzero output
differences from the unmodified-source legacy reference. It must not be
described as numerically equivalent or substituted for this official-reference
artifact. The local GPU rewrite is also not used for this EMISAC baseline.

## Scope of the comparison

- Displayed data are local sections 777–808 inclusive, a 32-section window.
  EMISAC was given that 32-section input. Other packaged outputs retain their
  recorded processing context; a common displayed crop does not prove a
  common original input extent.
- The original FECNet project targets RGB exposure correction. The comparison
  used a one-channel adaptation and task-specific fine-tuning. Its Bumbarger
  checkpoint must be available separately before readers can rerun that exact
  branch from weights; the demo does not replace it with unrelated weights.
- SSCOR learns from each test volume. The packaged Bumbarger comparison used
  200 epochs; the quick notebook does not repeat that training.
- vEMDestripe can run its released checkpoint live when enabled with a GPU.
  If a packaged fallback is selected, it is labelled as precomputed.
- Raw algorithm output is the default display. Optional global affine
  matching uses one transform per volume and is labelled separately, including
  in metric exports. It is not part of the unchanged algorithm output.
- Bumbarger13 has no paired clean target. The demo therefore omits PSNR/SSIM;
  its 32-section metrics must not be presented as the paper's full evaluation.

Dataset attribution is documented in [DATA_LICENSE.md](DATA_LICENSE.md).
