# vEMDestripe

Reference-guided restoration of axial stripe artifacts in volume electron microscopy (vEM).

This repository is the self-contained paper implementation of **vEMDestripe**. It includes synthetic paired-data generation, the fixed 1.91 M-parameter network, the fixed training objective, native/tiled inference, publication-oriented evaluation, and Slurm scripts for the complete workflow.

> Reproducibility note: the selected paper checkpoint was independently trained with 16×16 block averaging. Its scripts use up to 16 reference sections, whereas the current manuscript draft states 10. The default scripts reproduce the selected checkpoint. Read [docs/PAPER_CODE_ALIGNMENT.md](docs/PAPER_CODE_ALIGNMENT.md) before reporting results.

## Method at a glance

For every stripe-affected XY section, vEMDestripe concatenates the section with a structure-suppressed volume reference. The reference is obtained by floating-point averaging of screened sections followed by non-overlapping block averaging. A three-level encoder-decoder with nine channel-spatial gated (CSG) units predicts the restored section. Every CSG uses the identity-preserving response

```text
output = input + gamma * gated(input),  gamma = 0 at initialization.
```

Training minimizes

```text
L = L1 + 0.1 Ledge + 0.3 Lphoto + 0.1 LSSIM,
```

where `Lphoto` matches global and local mean/standard-deviation statistics at 32, 64, and 128 pixel scales. See [docs/METHOD.md](docs/METHOD.md) for the exact implementation.

## Run the complete-pipeline comparison in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-happy-802/vEMDestripe/blob/main/colab/vEMDestripe_Bumbarger_Pipeline_Demo.ipynb)

The default demo applies vEMDestripe's recorded **whole-volume reference
mean/std calibration before saving final output**, then evaluates the saved
files. Only vEMDestripe includes this stage; delivered EMISAC/FECNet/SSCOR
pixels are not changed. The evaluator does not normalize any method.

The published Bumbarger mapping uses all 2651 network-output sections and
11 original reference snapshots, then crops local Z 777–808 for display.
Optional live GPU mode fits a separate complete 32-section demo volume and
is labelled accordingly. EMISAC remains the verified unchanged-official-source
legacy-environment precomputed result. See [colab/README.md](colab/README.md).
This real dataset has no paired clean target, and demo metrics are not the
full-paper benchmark. The previous v2 notebook remains historical.

## Repository layout

```text
vEMDestripe/
├── colab/                 click-through Bumbarger13 comparison notebook
├── data_generation/       paired simulation and reference generation
├── vemdestripe/            model, loss, and shared reference operations
├── evaluation/             saved-output full-/no-reference benchmark
├── scripts/                generation, training, inference, evaluation Slurm jobs
├── tests/                  lightweight reproducibility tests
├── prepare_training_data.py public data-generation convenience entry point
├── train.py                fixed training entry point
└── infer.py                directory or TIFF inference entry point
```

## Installation

Python 3.10 or newer and PyTorch 2.x are recommended. On a CUDA server, install the PyTorch build appropriate for the installed driver, then install this project:

```bash
git clone https://github.com/simple-happy-802/vEMDestripe.git vEMDestripe
cd vEMDestripe
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Alternatively, adapt `environment.yml` to the CUDA version available on the cluster.

## Data preparation

Place clean training volumes as TIFF stacks under `data/clean_tifs/`:

```text
data/clean_tifs/
├── volume_a.tif
└── volume_b.tiff
```

Generate paired data locally:

```bash
python -m data_generation.generate \
  --input_dir data/clean_tifs \
  --output_dir data/training \
  --split all --image_size 1024 --format png \
  --val_ratio 0.10 --test_ratio 0.10 \
  --reference_num 50 --keep_clean_ratio 0.15 --seed 42 \
  --weights 0.22 0.09 0.11 0.07 0.13 0.09 0.09 0.14 0.06 0.00
```

This is the packaged form of `prepare_training_data_v_3_oblique_boundary.py`; the simulation equations and sampling behavior are preserved. Full layout and safety checks are in [docs/DATA.md](docs/DATA.md).

## Training

Single-node, two-GPU training:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data_dir data/training --image_size 1024 \
  --batch_size 2 --epochs 200 --lr 2e-4 --weight_decay 0.01 \
  --checkpoint_dir checkpoints/vemdestripe_retrain_seed42 \
  --output_dir outputs/vemdestripe_retrain_seed42 \
  --log_dir logs/tensorboard/vemdestripe_retrain_seed42 \
  --lambda_l1 1.0 --lambda_edge 0.1 --lambda_photo 0.3 \
  --lambda_ssim 0.1 --photo_std_weight 0.5 \
  --local_window_sizes 32,64,128 \
  --use_fused_reference --reference_mode fused \
  --reference_preprocess block_mean --reference_block_size 16 \
  --seed 42 --amp
```

The Slurm equivalent is `scripts/02_train.sbatch`. It requests two A100 GPUs, writes retraining runs to a directory separate from the included paper checkpoint, and resumes automatically from a full `latest.pth`/`best.pth` training checkpoint when present.

## Inference

Prepare one screened reference directory per volume:

```text
test_data/volume_a/*.png
references/volume_a/*.png
```

Then run:

```bash
python infer.py \
  --input_dir test_data/volume_a \
  --output_dir results/vemdestripe/volume_a \
  --model checkpoints/vemdestripe_seed42/best.pth \
  --reference references/volume_a \
  --num_reference 16 --reference_mode fused \
  --reference_preprocess block_mean --struct_block_size 16 \
  --reference_selection uniform \
  --image_size 1024 --inference_size 1024 --overlap_ratio 0.25 \
  --no_z_context --no_z_postprocess --no_preview
```

Directory inference now runs the complete pipeline by default. Network-only
uint8 predictions are retained in `output_dir/network_raw/`; one mean/std map
is fitted over **every output voxel** to the pooled pixels of the selected
**original reference snapshots**, not `reference.png`. Calibrated final
`*_deflickered.png` images and `CALIBRATION.json` are saved in `output_dir/`.
Evaluation must read that final directory and must not normalize again.
Use a new output directory; existing results are not overwritten.

For a network-only ablation, explicitly pass `--no_photometric_calibration`.
Do not use `--max_images` with full-volume calibration. Single-image input
requires the raw-only flag because it cannot establish full-volume statistics.
For already saved full-volume predictions, run the independent export stage:

```bash
python calibrate_volume.py --input-dir /path/to/all_network_outputs \
  --reference-dir /path/to/original_reference_snapshots \
  --reference-metadata /path/to/processing_info.json \
  --output-dir /path/to/new_final_outputs --output-stack /path/to/final_volume.tif
```

Use the exact recorded reference selection when metadata is available. The
explicit synthetic-only `--target-clean-volume clean.tif` option reproduces
privileged clean-target calibration; it is not an unlabeled deployment claim.

Exactly 1024×1024 inputs use one whole-image pass. Other sizes use reflection-padded, 25%-overlap tiled inference; exterior blend weights are forced to one so the cosine window cannot create a one-pixel black border.

For paired benchmarks, `scripts/03_infer.sbatch` can fall back to `<dataset>/groundtruth` or `<dataset>/test/clean` as the clean reference pool. That fallback is only for reproducing paired experiments. Real-data evaluation must use independently screened sections under `references/<dataset>`.

## Slurm workflow

Edit the partition/QoS lines if your cluster differs, create the expected folders, then submit all stages with dependencies:

```bash
mkdir -p logs data/clean_tifs test_data references
bash scripts/submit_pipeline.sh
```

If `data/training` already exists:

```bash
SKIP_DATA=true bash scripts/submit_pipeline.sh
```

The dependency workflow passes `checkpoints/vemdestripe_retrain_seed42/best.pth` from the training job to inference. To evaluate only the bundled paper checkpoint, submit `scripts/03_infer.sbatch` directly instead.

Individual stages can be submitted with environment overrides:

```bash
sbatch --export=ALL,DATA_DIR=/path/to/training scripts/02_train.sbatch
sbatch --export=ALL,MODEL_PATH=/path/to/best.pth,TEST_DATA_ROOT=/path/to/test_data scripts/03_infer.sbatch
sbatch --export=ALL,DATASETS=EPFL_center1024 scripts/04_evaluate.sbatch
```

## Evaluation

The benchmark reads saved final outputs without fitting or applying intensity normalization. It reports full-reference metrics when aligned clean targets are available and axial/no-reference metrics for real volumes. It also records per-section curves and evaluation-window provenance. See [docs/EVALUATION.md](docs/EVALUATION.md).

```bash
python -m evaluation.evaluate \
  --output-dir evaluation_output/vemdestripe \
  --evaluation-window-mode max-hf-mad --display-mode raw
```

## Checkpoints and reproducibility

Development checkpoints load strictly into `VEMDestripe` without key conversion. A compatibility test in this release verified all 205 state-dict keys and all 1,909,811 parameters. See [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Run the lightweight tests with:

```bash
python -m pytest -q
```

## Citation

Citation metadata is provided in `CITATION.cff`. Update it with the final journal, DOI, and release tag after publication.

## License

No software license was selected by the authors in the supplied materials. Choose and add an OSI-approved license before making the repository public; see [LICENSE_SELECTION_REQUIRED.md](LICENSE_SELECTION_REQUIRED.md). Until then, public reuse rights are not granted by this package.
