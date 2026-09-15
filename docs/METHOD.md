# Method specification

## Task definition

vEMDestripe corrects section-to-section brightness and contrast variation that appears as bands normal to the acquisition axis in XZ and YZ reslices. It is a two-input, two-dimensional restoration model: the first input channel is the current stripe-affected XY section and the second is a volume-level reference. Explicit neighboring-Z tensors and post-hoc Z smoothing are not part of the released network.

## Reference construction

Let the screened reference sections be \(R_1,\ldots,R_K\). Their floating-point mean is

\[
\bar R(x,y)=\frac{1}{K}\sum_{k=1}^{K}R_k(x,y).
\]

For each non-overlapping \(b\times b\) block \(\Omega\), the structure-suppressed reference is

\[
R_{\mathrm{block}}(x,y)=\frac{1}{|\Omega|}\sum_{(u,v)\in\Omega}\bar R(u,v),
\qquad (x,y)\in\Omega.
\]

The selected paper-checkpoint profile uses `K=16` and `b=16`. Averaging is performed in `float32` before quantization. Screening must be independent of model output and evaluation labels for a deployment or real-data claim.

## Network

The current section and reference are normalized to `[0,1]`, concatenated into a two-channel tensor, and processed by a three-level encoder-decoder:

- stem width: 32 channels;
- encoder widths: 64, 128, and 256 channels;
- bottleneck: two feature-transform blocks at 256 channels;
- decoder widths: 128, 64, and 32 channels;
- skip connections: additive, with bilinear interpolation (`align_corners=True`);
- output: direct one-channel restoration, not an input-plus-residual head;
- trainable parameters: 1,909,811.

There are nine channel-spatial gated units: one in the stem, three in the encoder, two at the bottleneck, and three in the decoder.

### Channel-spatial gated unit

For feature tensor \(F\), channel gating first computes a global-average descriptor and maps it through two 1×1 convolutions with a reduction ratio of four. Spatial gating then maps the channel-gated feature through two additional 1×1 convolutions to obtain a single spatial mask. The corrected residual implementation is

\[
F_{\mathrm{out}}=F+\gamma\,A_s(A_c(F)), \qquad \gamma_0=0.
\]

Therefore, each gate is exactly an identity mapping at initialization. This correction is the defining architectural change incorporated into the released method.

Each feature-transform block contains a GroupNorm-CSG branch and a GroupNorm-pointwise-feed-forward branch, both controlled by separate zero-initialized residual scales.

## Objective

For prediction \(\hat I\) and clean target \(I\), training minimizes

\[
\mathcal L=\mathcal L_1+0.1\mathcal L_{\mathrm{edge}}+0.3\mathcal L_{\mathrm{photo}}+0.1\mathcal L_{\mathrm{SSIM}}.
\]

`Ledge` is an L1 distance between Sobel gradient magnitudes. The photometric term compares per-image mean and standard deviation globally and over non-overlapping windows of 32, 64, and 128 pixels. Within every scale, the standard-deviation component has relative weight 0.5. `LSSIM` is `1 - SSIM` with an 11×11 Gaussian window where the image size permits it. Moment and SSIM calculations are forced to FP32 under AMP.

## Optimization

- optimizer: AdamW;
- initial learning rate: `2e-4`;
- weight decay: `0.01`;
- schedule: cosine annealing with warm restarts, `T0=30`, `Tmult=2`, `eta_min=2e-6`;
- training length: 200 epochs;
- seed: 42;
- distributed setup: two GPUs, per-GPU batch size 2;
- mixed precision: enabled on CUDA.

## Native and tiled inference

An exact 1024×1024 input with a 1024-pixel inference window bypasses tiling. Other sizes are reflection padded and processed using overlapping 1024×1024 windows. With the default overlap ratio of 0.25, interior overlaps use cosine weights, while weights on the exterior image boundary are set to one. The latter prevents zero-weight border pixels and the corresponding one-pixel black-edge artifact.
