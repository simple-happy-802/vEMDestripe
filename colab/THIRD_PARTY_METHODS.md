# Third-party methods in the Bumbarger13 demo

The comparison notebook keeps third-party code separate from vEMDestripe. It
links to the authors' repositories, runs EMISAC directly from the authors'
official source, and uses frozen outputs only where rerunning the method would
require missing task-specific weights or hours of dataset-specific training.
The frozen outputs are demonstration data, not code copied from repositories.

| Method | Public source | License status | Demo provenance |
|---|---|---|---|
| EMISAC | <https://github.com/azadis/EMISAC> | GPL-3.0 | Cloned and run live from official commit `7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5` on the 32-section demo window. |
| FECNet | <https://github.com/KevinJ-Huang/FECNet> | No license file was visible in the public repository when this package was prepared. Do not redistribute its source without permission. | Frozen output from the grayscale EM adaptation and Bumbarger-specific fine-tuning used in the comparison. |
| SSCOR | <https://github.com/AI-XPM-group/SSCOR> | Apache-2.0 in the checked repository | Frozen output from dataset-specific self-supervised training, seed 42. |

## EMISAC compatibility policy

The upstream EMISAC scripts target Python 2 and historical NumPy, SciPy,
scikit-image, Pillow, and h5py APIs. The notebook first verifies the pinned Git
commit, then writes a temporary Python-3-compatible working copy. Its patch is
shown in the notebook and is limited to:

- Python 2 print/iteration syntax;
- imports and argument names removed by current dependencies;
- deprecated NumPy/h5py aliases;
- integer divisions that were integer-valued under Python 2; and
- invoking the child script with the active Python interpreter.

The pinned Git commit and history remain unchanged; the two legacy scripts in
its disposable working tree receive only the compatibility edits listed above.
The objective, processing stages, default algorithm, and experiment parameters
are not replaced by the local GPU rewrite. The notebook records the source URL,
commit, pre-/post-patch source hashes, compatibility-manifest hash, and executed
command with the result.

## Important reproducibility limits

- The live EMISAC demonstration is intentionally limited to the same 32
  sections shown by the notebook. A complete 2,000-section optimization is too
  expensive for a quick free-Colab demonstration and is not implied by this
  run.
- The original FECNet project targets RGB exposure correction. The paper
  comparison used a one-channel adaptation with dataset-specific fine-tuning.
  Its frozen Bumbarger checkpoint must be released separately before claiming
  that readers can rerun this branch from weights.
- SSCOR learns from each test volume. The Bumbarger run used 200 epochs;
  training it inside the quick notebook would take hours and would no longer
  be a click-through demonstration.
- The notebook therefore runs official EMISAC and vEMDestripe live. FECNet and
  SSCOR remain immutable comparison artifacts with explicit provenance; the
  notebook does not pretend to rerun a different or unavailable configuration.
