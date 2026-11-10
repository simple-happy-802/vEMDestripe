# Optimized inference / 优化推理

`infer_optimized.py` runs the fixed E1 workflow. It screens the input volume, picks one section per group of eligible sections for 16 references, and builds a shared reference using float32 fusion and 32 × 32 block means. It runs inference and postprocessing on the native grid.

Install the [project dependencies](../README.md#installation). Place one volume of single-channel 8-bit PNG sections directly in a folder. Use a consistent numeric filename pattern, such as `slice_00001.png`, in Z order. All sections must have the same dimensions, at least 32 pixels and divisible by eight on each side. The reference selector needs at least 16 usable sections. Supply the complete volume for each run.

```bash
python infer_optimized.py --input-dir input/volume_a --output-dir results/volume_a --device cuda
```

Use `--device cpu` for a CPU run. The default checkpoint is the bundled portable E1 export. `--checkpoint` accepts another copy of the same verified file. See the [checkpoint record](../checkpoints/README.md) for its origin and [SHA-256](../checkpoints/CHECKSUMS.sha256) for the released checksum.

Choose a fresh output directory. `restored/` contains the final lossless PNGs with the input filenames. `processing.json` records the input order and hashes, selected reference filenames, checkpoint hash, geometry, device and processing time. The timer includes checkpoint loading, reference selection, inference, postprocessing and saving the PNGs. Use the final images for evaluation.

The [original workflow](METHOD.md#original-inference-postprocessing) uses its own separately trained block16 weights. The [Bumbarger13 notebook](../colab/README.md) documents the source of its saved comparison assets.

## 中文

优化版使用仓库附带的固定 E1 权重。程序从输入体筛选 16 张参考切片，用浮点均值融合和 32 × 32 分块均值构建共享参考图，然后逐层推理并执行后处理。

将同一体数据的 8 位单通道灰度 PNG 放在一个目录，按同一命名规则编号。各层 XY 尺寸相同，每边至少 32 像素且是 8 的倍数。输入至少要有 16 张可用的参考切片。按上面的命令运行，CPU 环境用 `--device cpu`。

最终无损 PNG 保存在 `restored/`，文件名与输入一致。处理记录 `processing.json` 包含输入顺序、尺寸、参考文件、权重校验值和用时。请用最终 PNG 进行评估。ZIP 上传和截面查看见[网页演示](../demo/README.md)。
