# Data generation and directory conventions

## Training source data

`data_generation/generate.py` packages the supplied `prepare_training_data_v_3_oblique_boundary.py` workflow. It expects one or more clean grayscale TIFF volumes. Input volumes are treated as source material for paired simulation; they are not included in this repository.

```text
data/clean_tifs/
├── specimen_01.tif       # Z × H × W
└── specimen_02.tiff
```

The default Slurm job writes to `data/training/` and refuses to run when that destination is non-empty, preventing accidental mixing of two simulations.

## Generated layout

For each source volume, the generator creates:

```text
data/training/<volume_name>/
├── train/
│   ├── clean/
│   ├── striped/
│   └── reference/
├── val/
│   ├── clean/
│   ├── striped/
│   └── reference/
└── test/
    ├── clean/
    ├── striped/
    └── reference/
```

Clean and striped images have identical basenames. Each reference directory contains individual selected sections and a `<split>_reference_fused.png` file. The generator first samples `reference_num=50` candidates from the complete volume and then retains candidates belonging to each split, capped at `reference_num // 3` per split. Consequently, 50 does **not** guarantee 16 references in every split; it produces at most 16 and can yield fewer, especially in validation and test splits.

The generator preserves its original behavior: the stored fused reference is the `float32` mean of the selected split references followed by 32×32 block averaging and uint8 quantization. The selected paper model does **not** train from that stored 32×32 image. Its training loader reads the individual `*_ref_*` sections, recomputes their floating-point mean, and applies 16×16 block averaging (`reference_preprocess=block_mean`, `reference_block_size=16`). This is why the original generator can be retained while the final checkpoint remains a genuine 16×16 retrained model.

## Simulation profile

The supplied profile uses mode weights

```text
0.22 0.09 0.11 0.07 0.13 0.09 0.09 0.14 0.06 0.00
```

for modes M0–M9, `keep_clean_ratio=0.15`, image size 1024, 10% validation, 10% test, and seed 42. The implementation includes the supplied global, local, oblique-boundary, and photometric perturbations. `dataset_info.json` records the realized output counts and paths.

## Test data

The batch-inference script accepts any of these per-dataset input layouts:

```text
test_data/<dataset>/*.png
test_data/<dataset>/input/*.png
test_data/<dataset>/striped/*.png
test_data/<dataset>/test/striped/*.png
```

For deployment or real-data evaluation, place manually screened reference sections in:

```text
references/<dataset>/*.png
```

The default reference priority is:

1. `references/<dataset>`;
2. `test_data/<dataset>/reference`;
3. `test_data/<dataset>/groundtruth` (paired benchmark only);
4. `test_data/<dataset>/test/clean` (paired benchmark only).

Using paired targets as the reference pool is allowed only to reproduce a paired benchmark protocol. It must be disclosed, and no target slice being scored should be selected as an independent reference unless the experimental design explicitly permits this. For a defensible real-data claim, screen reference sections without consulting model outputs or unavailable clean labels and save their filenames as provenance.

## Data licensing and privacy

This code does not redistribute source vEM volumes. Before publishing links or derived examples, verify each dataset's original license, attribution requirements, and any restrictions on redistribution. Do not commit large raw volumes, generated training sets, checkpoints, or result stacks to Git; the supplied `.gitignore` excludes them by default.
