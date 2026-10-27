# vEMDestripe 中文使用说明

本目录是论文方法 **vEMDestripe** 的独立开源整理版，包含训练数据生成、固定网络结构、固定损失函数、训练、推理、批量评估和 Slurm 脚本。公开代码中不再使用开发阶段实验编号作为算法名称。

## Google Colab 完整流程对比演示

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Pipeline_Demo.ipynb)

新版演示为：**网络输出 → 整卷均值／标准差匹配到固定 Original `slice_00793.png` → 保存最终图像 → 评估**。
这一步只属于 vEMDestripe；EMISAC、FECNet、SSCOR 不额外校准。评估程序只读最终文件，不归一化。
默认源统计仍使用完整 2651 张网络输出，目标只取用户指定的 Original-793 全部像素，
整卷应用同一映射后截取 32 张展示。不要求 GPU；可选现场模式使用全部 32 张新输出
作为源统计，目标仍固定为 Original-793，并明确区分这两个上下文。
这次只改演示目标，不修改训练或正式 `infer.py` 的默认原始参考池校准流程；
亮度／对比度匹配不是空间配准，也不冒充此前论文参考池协议。旧版本标签保留。
EMISAC 仍是未改官方源码在固定旧环境生成并验证的结果。详见 [colab/README.md](colab/README.md)。

## 最重要的复现说明

项目已将最终选择的论文权重导出到 `checkpoints/vemdestripe_seed42/best.pth`。该权重是使用 16×16 参考图分块独立重训得到的，不是把 32×32 权重在推理时临时改成 16×16。对应推理脚本从参考目录均匀选择最多 16 张切片，先用 `float32` 求均值，再进行 16×16 非重叠分块均值。当前论文草稿写的是 10 张参考切片，因此仍需统一参考切片数量；详见 [docs/PAPER_CODE_ALIGNMENT.md](docs/PAPER_CODE_ALIGNMENT.md)。

## 服务器快速运行

将整个 `vEMDestripe` 文件夹上传到服务器，例如：

```text
/path/to/vEMDestripe
```

进入项目并创建日志目录：

```bash
cd /path/to/vEMDestripe
mkdir -p logs data/clean_tifs test_data references
```

将用于合成训练数据的无条纹 TIFF 体数据放入 `data/clean_tifs/`。如果手动运行，请确认前一个任务成功结束后再提交下一个任务：

```bash
sbatch scripts/01_generate_data.sbatch
sbatch scripts/02_train.sbatch
sbatch scripts/03_infer.sbatch
sbatch scripts/04_evaluate.sbatch
```

不要把上面四条命令一次性连续提交，因为它们本身没有 Slurm 依赖关系。更推荐使用自动依赖脚本：

```bash
bash scripts/submit_pipeline.sh
```

已有训练数据时跳过生成：

```bash
SKIP_DATA=true DATA_DIR=/你的训练数据路径 bash scripts/submit_pipeline.sh
```

一键流程会把本次重训得到的 `checkpoints/vemdestripe_retrain_seed42/best.pth` 自动传给推理任务。若只想评估项目内已经附带的论文权重，请直接提交 `scripts/03_infer.sbatch`，不要运行重训流水线。

只使用已有权重推理：

```bash
sbatch --export=ALL,MODEL_PATH=/path/to/best.pth,TEST_DATA_ROOT=/path/to/test_data scripts/03_infer.sbatch
```

如果测试集是 `test_medium`：

```bash
sbatch --export=ALL,MODEL_PATH=/path/to/best.pth,TEST_DATA_ROOT=/path/to/test_medium scripts/03_infer.sbatch
```

每个真实数据集都应在 `references/<数据集名>/` 中放置独立人工筛选的正常参考切片。对于带配对标签的模拟测试集，脚本允许使用 `groundtruth/` 作为复现实验用参考池，但论文中必须明确说明，且不能把这种结果当作无标签真实场景性能。

训练与推理细节分别见 [docs/DATA.md](docs/DATA.md)、[docs/METHOD.md](docs/METHOD.md) 和 [docs/EVALUATION.md](docs/EVALUATION.md)。

## 最终输出和评估分离

新版 `infer.py` 目录模式默认保留 `network_raw/` 中间结果，并把校准后的最终
`*_deflickered.png` 保存到指定输出目录根部，同时记录 `CALIBRATION.json`。
源统计来自全部输出像素，目标来自本次实际选择的原始参考切片像素池，不是融合图或分块图。
每片应用同一张 256 级映射表；不能逐切片拟合，也不要只用评估窗口重新拟合。

已生成的完整网络输出可以单独校准导出：

```bash
python calibrate_volume.py --input-dir /path/to/all_network_outputs \
  --reference-dir /path/to/original_reference_snapshots \
  --reference-metadata /path/to/processing_info.json \
  --output-dir /path/to/new_final_outputs --output-stack /path/to/final.tif
```

`python -m evaluation.evaluate` 和 Slurm 评估脚本只计算已保存文件的指标，不做亮度匹配。
其他方法保持原交付像素。网络原始输出消融请显式指定 `--no_photometric_calibration`。
不要在默认整卷校准下指定 `--max_images`。输出目录已有结果时请改用新目录。
