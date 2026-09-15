# Checkpoints

The selected 16×16 paper weight is included at `checkpoints/vemdestripe_seed42/best.pth`. It is a portable inference export of the independently trained seed-42 checkpoint: model tensors are unchanged, while optimizer state, absolute server paths, and development-only labels are omitted. Override `MODEL_PATH` in the Slurm scripts to use another checkpoint.

SHA-256 is recorded in `CHECKSUMS.sha256`. Other binary checkpoints remain ignored by Git.

