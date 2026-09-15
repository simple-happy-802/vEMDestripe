# Manuscript–code alignment audit

This audit records the reproducibility discrepancy found while packaging the supplied method. It should be resolved before the final manuscript, result tables, checkpoint release, and GitHub tag are frozen.

| Item | Selected paper checkpoint/scripts | Current manuscript draft reviewed 2026-09-01 |
|---|---:|---:|
| Inference reference sections | up to 16 | 10 |
| Reference block size | 16×16 | 16×16 |
| Reference fusion order | float mean, then block mean | float mean, then block mean |
| Model | fixed identity-residual CSG network | same conceptual network |
| Loss | L1 + 0.1 edge + 0.3 photo + 0.1 SSIM | same |

The release defaults to **up to 16 references and 16×16 blocks**. The block-size setting is backed by its own independently trained checkpoint. Inspection of that checkpoint confirmed `reference_preprocess=block_mean`, `reference_block_size=16`, seed 42, fixed model, and fixed objective. Therefore, 16×16 is not an inference-only substitution applied to a 32×32 model.

The training generator receives `reference_num=50`, samples candidates globally, and caps each split at 16. The realized count can be below 16 because only candidates falling inside a split are retained. Inference independently selects up to 16 sections from its reference directory. These facts should be stated precisely instead of saying that training always used exactly 16 references.

The remaining mismatch concerns the number of inference reference sections. Before publication, choose one of the following evidence-based resolutions:

1. **Checkpoint-script-faithful route:** retain the selected 16×16 weight, rerun all reported inference/evaluation with up to 16 references, and revise the manuscript reference count to 16.
2. **Manuscript-faithful inference route:** retain the independently trained 16×16 weight, rerun every reported test with exactly 10 screened references, verify that this is the intended inference protocol, and regenerate every headline table and figure. If reference-count conditioning materially changes performance, also retrain under a strictly controlled 10-reference protocol.

Do not report results produced with 16 references while claiming 10. The block-size ablation correctly uses separately retrained weights for no block, 16×16, 32×32, and 64×64; preserve that mapping in any paper table.

The legacy development checkpoint is structurally compatible with `VEMDestripe`: all 205 state-dict keys match strictly and the parameter count is 1,909,811. The public method name is **vEMDestripe**; development experiment labels should appear only in private migration records, not as the algorithm name in the paper or README.
