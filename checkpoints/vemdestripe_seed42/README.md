# vEMDestripe seed-42 checkpoint

`best.pth` contains the exact model tensors selected from the independently trained 16×16 reference-block run.

```text
architecture:            vEMDestripe
loss:                    L1 + 0.1 edge + 0.3 photo + 0.1 SSIM
reference preprocessing: float mean, then 16x16 block averaging
inference reference cap: 16
seed:                    42
selected epoch:          146
parameters:              1,909,811
state-dict entries:      205
```

The portable file excludes optimizer and scheduler state and is intended for inference. Its SHA-256 is listed in the parent `CHECKSUMS.sha256`.

