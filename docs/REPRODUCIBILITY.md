# Reproducibility checklist

Record the following with every reported run:

- Git commit and release tag;
- checkpoint SHA-256, training seed, and checkpoint epoch;
- source dataset names, versions, licenses, and exact train/validation/test split manifests;
- realized numbers of clean, corrupted, and reference images in `dataset_info.json`;
- simulation weights, `keep_clean_ratio`, image size, and generator seed;
- number and filenames of screened reference sections;
- reference selection rule, fusion dtype, and block size;
- whether targets or ground-truth sections were available to reference selection;
- input bit depth and any conversion to uint8;
- full-image or tiled inference, window size, overlap, padding, and blend weights;
- GPU model, CUDA, PyTorch, Python, and dependency versions;
- evaluation window, excluded borders, resampling rules, and metric implementation version.

## Determinism

The scripts seed Python, NumPy, and PyTorch with 42. DDP ranks receive `seed + rank` for independent augmentation streams. This improves repeatability but does not guarantee bitwise identity across different CUDA/cuDNN or PyTorch releases. Report mean and variability across independent training seeds for claims about model superiority.

## Release verification

Before tagging a release:

```bash
python -m compileall -q .
python -m pytest -q
python train.py --help
python infer.py --help
python -m data_generation.generate --help
python -m evaluation.evaluate --help
```

Then perform one end-to-end smoke run with a small TIFF and `MAX_SAMPLES`, check that a checkpoint reloads strictly, verify that 1024×1024 inference reports no tiling, and inspect the first/last rows and columns of every restored image for border artifacts.

