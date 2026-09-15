# Checkpoint compatibility

The public `VEMDestripe` module preserves the submodule names and tensor shapes used by the supplied development checkpoint. A strict compatibility check gave:

```text
trainable parameters: 1,909,811
state-dict entries:    205
missing keys:          0
unexpected keys:       0
shape mismatches:      0
```

Both raw state dictionaries and trainer checkpoints containing `model_state_dict` are accepted by `infer.py`. Checkpoints saved under `DistributedDataParallel` with a leading `module.` prefix are normalized before strict loading.

The selected 16×16 training checkpoint was verified with:

```text
epoch: 146
reference_preprocess: block_mean
reference_block_size: 16
seed: 42
```

For the public repository, place the same binary at:

```text
checkpoints/vemdestripe_seed42/best.pth
```

For public archival, publish the file's SHA-256 and the complete training configuration. Do not rename a checkpoint from a different architecture and present it as vEMDestripe merely because non-strict loading succeeds; inference deliberately uses strict loading to prevent that error.
