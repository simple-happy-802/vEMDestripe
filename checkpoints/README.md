# Checkpoints

The selected 16×16 paper weight is included at `checkpoints/vemdestripe_seed42/best.pth`. It is a portable inference export of the independently trained seed-42 checkpoint: model tensors are unchanged, while optimizer state, absolute server paths, and development-only labels are omitted. Override `MODEL_PATH` in the Slurm scripts to use another checkpoint.

SHA-256 is recorded in `CHECKSUMS.sha256`. Training checkpoints are ignored by Git.


The optimized workflow uses `checkpoints/vemdestripe_e1_block32/best.pth`.
This is an inference export of the fixed E1 epoch-197 checkpoint
(`43a700e52ef72b21590bbd6ad069aa699f829d90abf27aa256aa2dd599e8a505`).
All 205 state tensors are preserved exactly. The export includes only model
tensors and basic provenance, and loads with PyTorch's weights-only mode.
