# vEMDestripe 中文使用说明

本目录是论文方法 **vEMDestripe** 的独立开源整理版，包含训练数据生成、固定网络结构、固定损失函数、训练、推理、批量评估和 Slurm 脚本。公开代码中不再使用开发阶段实验编号作为算法名称。

## Google Colab 完整流程对比演示

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Pipeline_Demo.ipynb)

演示会保存 vEMDestripe 推理后处理的结果，提供同位置 XY、XZ、YZ 查看、
对比指标和 ZIP 下载。默认模式用已保存的网络输出，在 CPU 上运行。
GPU 模式可现场推理 32 张切片。详见 [Colab 演示说明](colab/README.md)。

## 优化版推理

把同一体数据的 8 位灰度 PNG 按切片编号放在一个目录。仓库附带固定 E1 权重，
下方命令完成推理和后处理，并保存最终 PNG 和处理用时。

```bash
python infer_optimized.py --input-dir input/volume_a --output-dir results/volume_a --device cuda
```

在 CPU 上使用 `--device cpu`。详见 [优化推理说明](docs/OPTIMIZED_INFERENCE.md)
和 [网页演示](demo/README.md)。

以下训练及 Slurm 脚本对应仓库的原版 block16 流程。使用这些脚本复现实验时，
请核对参考切片数量与权重版本，见 [复现说明](docs/PAPER_CODE_ALIGNMENT.md)。

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

## 最终输出和评估

原版 `infer.py` 目录模式在推理后执行输出处理。最终 PNG 写入指定输出目录，
中间文件保存在 `network_raw/`。评估时读取最终文件。命令选项见
[原版推理说明](docs/METHOD.md#original-inference-postprocessing)。
