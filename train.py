"""Train vEMDestripe on paired simulated stripe/clean sections.

The public entry point exposes one fixed paper model and one fixed objective;
experimental architecture and loss switches are intentionally excluded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import cv2
import random
import json
import hashlib
from contextlib import nullcontext

from vemdestripe.model import VEMDestripe
from vemdestripe.loss import VEMDestripeLoss
from vemdestripe.reference import block_average

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_grayscale_image(path):
    """加载灰度图；OpenCV 失败时回退到 PIL（部分 PNG 仅 PIL 可读）。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img

    if PIL_AVAILABLE:
        try:
            with Image.open(path) as pil_img:
                return np.array(pil_img.convert('L'))
        except Exception:
            pass
    return None


def pair_clean_striped_files(clean_files, striped_files):
    """按文件名配对 clean/striped，避免列表长度不一致导致错位。"""
    striped_map = {os.path.basename(p): p for p in striped_files}
    paired_clean, paired_striped = [], []
    for clean_path in clean_files:
        striped_path = striped_map.get(os.path.basename(clean_path))
        if striped_path is not None:
            paired_clean.append(clean_path)
            paired_striped.append(striped_path)
    return paired_clean, paired_striped


def destroy_reference_structure(image, block_size=16):
    """Destroy spatial structure while exactly preserving the intensity histogram.

    ``block_size`` is retained only for backward-compatible command lines;
    the full-image pixel permutation deliberately preserves no anatomy at any
    spatial scale.
    """
    if image is None:
        return None
    if block_size < 1:
        raise ValueError('reference block_size 必须大于 0')
    source = np.ascontiguousarray(image)
    # The content-stable seed makes all DDP ranks construct the same reference.
    # The full permutation leaves no source position at its original location.
    seed = int.from_bytes(hashlib.blake2b(source.tobytes(), digest_size=8).digest(), 'little')
    rng = np.random.default_rng(seed)
    if source.ndim == 2:
        return source.reshape(-1)[rng.permutation(source.size)].reshape(source.shape).copy()
    if source.ndim == 3:
        height, width, channels = source.shape
        pixels = source.reshape(height * width, channels)
        return pixels[rng.permutation(height * width)].reshape(source.shape).copy()
    raise ValueError(f'Unsupported reference image dimensions: {source.shape}')


def block_mean_reference_structure(image, block_size=16):
    """Replace every non-overlapping block by its mean, matching V3 generation."""
    if image is None:
        return None
    if block_size < 1:
        raise ValueError('reference_block_size must be positive')
    result = block_average(image, block_size)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


class PairedVEMDataset(Dataset):
    """Paired clean/corrupted sections with one volume-level reference."""

    def __init__(self, data_dir, split='train', image_size=512, augment=True,
                 use_fused_reference=True, z_sequence_length=8,
                 reference_mode='fused',
                 single_reference_policy='random', single_reference_index=0,
                 reference_preprocess='block_mean', reference_block_size=16):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.use_fused_reference = use_fused_reference
        self.z_sequence_length = z_sequence_length
        self.reference_mode = reference_mode
        self.single_reference_policy = single_reference_policy
        self.single_reference_index = single_reference_index
        self.reference_preprocess = reference_preprocess
        self.reference_block_size = int(reference_block_size)
        if self.reference_mode not in ('fused', 'single'):
            raise ValueError(f'未知 reference_mode: {self.reference_mode}')
        if self.single_reference_policy not in ('random', 'fixed'):
            raise ValueError(f'未知 single_reference_policy: {self.single_reference_policy}')
        valid_reference_preprocess = ('stored', 'raw_mean', 'block_mean', 'zero', 'self')
        if self.reference_preprocess not in valid_reference_preprocess:
            raise ValueError(
                f'未知 reference_preprocess: {self.reference_preprocess}; '
                f'可选值为 {valid_reference_preprocess}'
            )
        if self.reference_block_size < 1:
            raise ValueError('reference_block_size must be positive')
        if self.reference_mode == 'single' and self.reference_preprocess not in ('stored', 'zero', 'self'):
            raise ValueError('raw_mean/block_mean 只适用于 reference_mode=fused')

        self.is_multi_dataset = self._detect_structure()

        if self.is_multi_dataset:
            self._load_multi_dataset()
        else:
            self._load_flat_dataset()

        self._compute_dataset_statistics()

        if len(self.clean_images) == 0:
            raise RuntimeError(
                f"数据集 [{split}] 无有效图像，请检查 {self.data_dir} 或运行 "
                f"python scripts/validate_training_images.py"
            )

        print(
            f"数据集 [{split}]: {len(self.clean_images)} 张图像 "
            f"(来自 {len(self.datasets)} 个数据集, reference_mode={self.reference_mode}, "
            f"reference_preprocess={self.reference_preprocess})"
        )

    def _load_single_references(self, reference_dir, fallback_path):
        """Load individual split_ref_XXX images without changing their structure."""
        candidates = sorted(glob(os.path.join(reference_dir, f'{self.split}_ref_*.*')))
        references = []
        for path in candidates:
            image = load_grayscale_image(path)
            if image is not None:
                references.append(image)
        if not references:
            fallback = load_grayscale_image(fallback_path)
            if fallback is not None:
                print(f"  [{self.split}] 警告: 未找到单参考图，回退为第一张 clean 图的结构破坏版本")
                references.append(fallback)
        return references

    def _detect_structure(self):
        subdirs = [d for d in os.listdir(self.data_dir)
                   if os.path.isdir(os.path.join(self.data_dir, d)) and d != 'dataset_info.json']

        if len(subdirs) > 0:
            first_subdir = os.path.join(self.data_dir, subdirs[0])
            if os.path.exists(os.path.join(first_subdir, 'train')):
                return True
        return False

    def _load_multi_dataset(self):
        self.datasets = []
        self.clean_images = []
        self.striped_images = []
        self.dataset_indices = []

        subdirs = [d for d in os.listdir(self.data_dir)
                   if os.path.isdir(os.path.join(self.data_dir, d)) and d != 'dataset_info.json']

        for ds_name in sorted(subdirs):
            ds_path = os.path.join(self.data_dir, ds_name)
            split_path = os.path.join(ds_path, self.split)

            if not os.path.exists(split_path):
                continue

            clean_dir = os.path.join(split_path, 'clean')
            striped_dir = os.path.join(split_path, 'striped')
            reference_dir = os.path.join(split_path, 'reference')

            clean_files = sorted(glob(os.path.join(clean_dir, '*g')))
            striped_files = sorted(glob(os.path.join(striped_dir, '*g')))

            clean_files, striped_files = pair_clean_striped_files(clean_files, striped_files)
            if len(clean_files) != len(striped_files):
                print(f"  [{ds_name}/{self.split}] 警告: clean/striped 配对后 {len(clean_files)} 对")

            if len(clean_files) == 0:
                continue

            fused_ref = None
            single_references = []
            if self.reference_mode == 'single' or self.reference_preprocess in ('raw_mean', 'block_mean'):
                single_references = self._load_single_references(reference_dir, clean_files[0])
            if self.reference_mode != 'single' and self.reference_preprocess == 'stored':
                fused_ref_path = os.path.join(reference_dir, f"{self.split}_reference_fused.png")
                if not os.path.exists(fused_ref_path):
                    fused_candidates = sorted(glob(os.path.join(reference_dir, f"{self.split}_reference_fused.*")))
                    fused_ref_path = fused_candidates[0] if fused_candidates else None
                fused_ref = load_grayscale_image(fused_ref_path) if fused_ref_path else None
                if fused_ref is None:
                    fused_ref = load_grayscale_image(clean_files[0])

            if self.reference_preprocess in ('raw_mean', 'block_mean'):
                fused_float = np.mean(
                    [reference.astype(np.float32) for reference in single_references],
                    axis=0,
                    dtype=np.float32,
                )
                if self.reference_preprocess == 'block_mean':
                    fused_ref = block_mean_reference_structure(
                        fused_float, self.reference_block_size
                    )
                else:
                    fused_ref = np.clip(fused_float, 0.0, 255.0).astype(np.uint8)

            self.datasets.append({
                'name': ds_name,
                'clean_files': clean_files,
                'striped_files': striped_files,
                'fused_reference': fused_ref,
                'single_references': single_references,
                'reference_dir': reference_dir,
                'start_idx': len(self.clean_images),
                'length': len(clean_files)
            })

            self.clean_images.extend(clean_files)
            self.striped_images.extend(striped_files)

            for _ in range(len(clean_files)):
                self.dataset_indices.append(len(self.datasets) - 1)

    def _load_flat_dataset(self):
        clean_dir = os.path.join(self.data_dir, self.split, 'clean')
        striped_dir = os.path.join(self.data_dir, self.split, 'striped')
        reference_dir = os.path.join(self.data_dir, self.split, 'reference')

        clean_files = sorted(glob(os.path.join(clean_dir, '*g')))
        striped_files = sorted(glob(os.path.join(striped_dir, '*g')))
        clean_files, striped_files = pair_clean_striped_files(clean_files, striped_files)

        self.datasets = [{
            'name': 'default',
            'clean_files': clean_files,
            'striped_files': striped_files,
            'fused_reference': None,
            'single_references': [],
            'reference_dir': reference_dir,
            'start_idx': 0,
            'length': len(clean_files)
        }]

        self.clean_images = clean_files
        self.striped_images = striped_files
        self.dataset_indices = [0] * len(clean_files)

        if self.reference_mode == 'single' or self.reference_preprocess in ('raw_mean', 'block_mean'):
            self.datasets[0]['single_references'] = self._load_single_references(reference_dir, clean_files[0])
        if self.reference_mode != 'single' and self.reference_preprocess == 'stored' and self.use_fused_reference:
            fused_ref_path = os.path.join(reference_dir, f"{self.split}_reference_fused.png")
            if not os.path.exists(fused_ref_path):
                fused_candidates = sorted(glob(os.path.join(reference_dir, f"{self.split}_reference_fused.*")))
                fused_ref_path = fused_candidates[0] if fused_candidates else None
            if fused_ref_path:
                self.datasets[0]['fused_reference'] = load_grayscale_image(fused_ref_path)

        if self.reference_preprocess in ('raw_mean', 'block_mean'):
            references = self.datasets[0]['single_references']
            fused_float = np.mean(
                [reference.astype(np.float32) for reference in references],
                axis=0,
                dtype=np.float32,
            )
            if self.reference_preprocess == 'block_mean':
                self.datasets[0]['fused_reference'] = block_mean_reference_structure(
                    fused_float, self.reference_block_size
                )
            else:
                self.datasets[0]['fused_reference'] = np.clip(
                    fused_float, 0.0, 255.0
                ).astype(np.uint8)

    def _compute_dataset_statistics(self):
        self.dataset_statistics = {}

        for ds_idx, ds in enumerate(self.datasets):
            indices = [i for i, d in enumerate(self.dataset_indices) if d == ds_idx]

            if len(indices) < 2:
                self.dataset_statistics[ds_idx] = {
                    'median_brightness': 0.5,
                    'median_contrast': 0.2,
                    'num_samples': len(indices)
                }
                continue

            brightness_list = []
            contrast_list = []

            sample_indices = random.sample(indices, min(100, len(indices)))

            for idx in sample_indices:
                img = load_grayscale_image(self.clean_images[idx])
                if img is not None:
                    img_float = img.astype(np.float32) / 255.0
                    brightness_list.append(np.mean(img_float))
                    contrast_list.append(np.std(img_float))

            if brightness_list:
                self.dataset_statistics[ds_idx] = {
                    'median_brightness': float(np.median(brightness_list)),
                    'median_contrast': float(np.median(contrast_list)),
                    'num_samples': len(indices)
                }
            else:
                self.dataset_statistics[ds_idx] = {
                    'median_brightness': 0.5,
                    'median_contrast': 0.2,
                    'num_samples': len(indices)
                }

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        for attempt in range(10):
            cur_idx = idx if attempt == 0 else random.randint(0, len(self.clean_images) - 1)
            ds_idx = self.dataset_indices[cur_idx]
            dataset = self.datasets[ds_idx]

            clean_img = load_grayscale_image(self.clean_images[cur_idx])
            striped_img = load_grayscale_image(self.striped_images[cur_idx])
            if clean_img is None or striped_img is None:
                if attempt == 9:
                    raise RuntimeError(
                        f"无法读取图像: {self.clean_images[cur_idx]} / {self.striped_images[cur_idx]}"
                    )
                continue

            clean_img = clean_img.astype(np.float32) / 255.0
            striped_img = striped_img.astype(np.float32) / 255.0
            idx = cur_idx
            break

        if self.reference_preprocess == 'zero':
            ref_img = np.zeros_like(striped_img, dtype=np.float32)
        elif self.reference_preprocess == 'self':
            ref_img = striped_img.copy()
        elif self.reference_mode == 'single' and dataset.get('single_references'):
            single_refs = dataset['single_references']
            if self.single_reference_policy == 'fixed':
                ref_img = single_refs[self.single_reference_index % len(single_refs)].copy().astype(np.float32) / 255.0
            else:
                ref_img = random.choice(single_refs).copy().astype(np.float32) / 255.0
        elif dataset['fused_reference'] is not None:
            ref_img = dataset['fused_reference'].copy().astype(np.float32) / 255.0
        else:
            ref_img = striped_img.copy()

        if clean_img.shape[0] != self.image_size:
            clean_img = cv2.resize(clean_img, (self.image_size, self.image_size))
            striped_img = cv2.resize(striped_img, (self.image_size, self.image_size))
            ref_img = cv2.resize(ref_img, (self.image_size, self.image_size))

        if self.augment:
            clean_img, striped_img, ref_img = self._augment(clean_img, striped_img, ref_img)

        clean_tensor = torch.from_numpy(clean_img).unsqueeze(0).float()
        striped_tensor = torch.from_numpy(striped_img).unsqueeze(0).float()
        ref_tensor = torch.from_numpy(ref_img).unsqueeze(0).float()

        stats = self.dataset_statistics.get(ds_idx, {
            'median_brightness': 0.5,
            'median_contrast': 0.2
        })

        return {
            'gt': clean_tensor,
            'input': striped_tensor,
            'reference': ref_tensor,
            'dataset': dataset['name'],
            'index': idx,
            'ds_index': ds_idx,
            'ds_brightness': stats['median_brightness'],
            'ds_contrast': stats['median_contrast']
        }

    def _augment(self, clean, striped, ref):
        if random.random() > 0.5:
            clean = np.fliplr(clean).copy()
            striped = np.fliplr(striped).copy()
            ref = np.fliplr(ref).copy()

        if random.random() > 0.5:
            clean = np.flipud(clean).copy()
            striped = np.flipud(striped).copy()
            ref = np.flipud(ref).copy()

        k = random.randint(0, 3)
        clean = np.rot90(clean, k).copy()
        striped = np.rot90(striped, k).copy()
        ref = np.rot90(ref, k).copy()

        return clean, striped, ref


class VEMDestripeTrainer:
    """Single-node or DDP trainer for the released vEMDestripe model."""

    def __init__(self, args):
        self.args = args
        self._setup_distributed()
        # Give each DDP rank an independent augmentation/data-loader RNG stream.
        rank_seed = args.seed + self.rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(rank_seed)

        if self.is_main_process:
            print("Model: vEMDestripe (CSG-gated encoder-decoder)")
        self.model = VEMDestripe(
            in_channels=2,
            out_channels=1,
            base_features=32,
            num_blocks=2,
        )
        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self.use_amp = bool(getattr(args, 'amp', False) and self.device.type == 'cuda')
        # torch.amp is the supported API in PyTorch 2.x; keep a fallback for
        # older server environments.
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.accumulation_steps = max(1, int(getattr(args, 'accumulation_steps', 1)))

        local_window_sizes = tuple(
            int(value.strip()) for value in args.local_window_sizes.split(',') if value.strip()
        )
        self.criterion = VEMDestripeLoss(
            lambda_l1=args.lambda_l1,
            lambda_edge=args.lambda_edge,
            lambda_photo=args.lambda_photo,
            lambda_ssim=args.lambda_ssim,
            photo_std_weight=args.photo_std_weight,
            local_window_sizes=local_window_sizes,
        ).to(self.device)
        if self.is_main_process:
            print(
                "Loss: L1 + edge + global/local photometric moments + SSIM "
                f"(criterion={self.criterion.__class__.__name__})"
            )

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=30,
            T_mult=2,
            eta_min=args.lr * 0.01
        )

        # 处理参考图像融合参数
        use_fused_ref = True
        if hasattr(args, 'no_fused_reference') and args.no_fused_reference:
            use_fused_ref = False
        elif hasattr(args, 'use_fused_reference') and args.use_fused_reference:
            use_fused_ref = True
        reference_mode = getattr(args, 'reference_mode', 'fused')

        train_dataset = PairedVEMDataset(
            data_dir=args.data_dir,
            split='train',
            image_size=args.image_size,
            augment=True,
            use_fused_reference=use_fused_ref,
            reference_mode=reference_mode,
            single_reference_policy=args.single_reference_policy,
            single_reference_index=args.single_reference_index,
            reference_preprocess=args.reference_preprocess,
            reference_block_size=args.reference_block_size,
        )
        val_dataset = PairedVEMDataset(
            data_dir=args.data_dir,
            split='val',
            image_size=args.image_size,
            augment=False,
            use_fused_reference=use_fused_ref,
            reference_mode=reference_mode,
            single_reference_policy=args.single_reference_policy,
            single_reference_index=args.single_reference_index,
            reference_preprocess=args.reference_preprocess,
            reference_block_size=args.reference_block_size,
        )
        self.train_sampler = DistributedSampler(
            train_dataset, shuffle=True, seed=args.seed
        ) if self.distributed else None
        self.val_sampler = DistributedSampler(
            val_dataset, shuffle=False, seed=args.seed
        ) if self.distributed else None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        if self.is_main_process:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            os.makedirs(args.output_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=args.log_dir)
        else:
            self.writer = None
        self._barrier()

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self._resume_if_requested()
        self._barrier()

        self._print_model_info()

    def _setup_distributed(self):
        """Auto-enable single-node DDP when launched through torchrun."""
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.distributed = self.world_size > 1
        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError('DDP training requires CUDA GPUs.')
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.rank = int(os.environ['RANK'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.rank = 0
            self.local_rank = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_main_process = self.rank == 0

    def _barrier(self):
        if self.distributed:
            dist.barrier()

    def _reduce_epoch_losses(self, losses, num_batches):
        """Report rank-averaged losses instead of rank-0's data shard only."""
        if not self.distributed:
            return {key: value / max(1, num_batches) for key, value in losses.items()}
        keys = list(losses.keys())
        values = [losses[key] for key in keys] + [float(num_batches)]
        reduced = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total_batches = max(1.0, reduced[-1].item())
        return {key: reduced[index].item() / total_batches for index, key in enumerate(keys)}

    def _autocast_context(self):
        """Use Tensor Cores on CUDA when --amp is selected."""
        if self.use_amp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    def _resume_if_requested(self):
        """Restore model, optimizer, scheduler, and AMP state."""
        resume_path = getattr(self.args, 'resume', None)
        if not resume_path:
            return
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f'Resume checkpoint does not exist: {resume_path}')

        try:
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location='cpu')

        model_for_load = self.model.module if self.distributed else self.model
        model_for_load.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # map_location='cpu' is safer for loading, then move optimizer
            # moments to the local DDP rank before the next optimizer step.
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)

        completed_epoch = int(checkpoint.get('epoch', 0))
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif completed_epoch > 0:
            # Older checkpoints did not store scheduler state. Reconstruct the
            # epoch-level WarmRestarts position without changing model weights.
            self.scheduler.step(completed_epoch)
        scaler_state = checkpoint.get('grad_scaler_state_dict')
        if scaler_state:
            self.grad_scaler.load_state_dict(scaler_state)

        self.start_epoch = completed_epoch + 1
        self.best_val_loss = float(checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf'))))
        # Legacy latest.pth files did not record the historical best. Recover
        # it from best.pth so resuming cannot replace a genuinely better model.
        if 'best_val_loss' not in checkpoint:
            best_path = os.path.join(os.path.dirname(resume_path), 'best.pth')
            if os.path.isfile(best_path) and os.path.abspath(best_path) != os.path.abspath(resume_path):
                try:
                    best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                except TypeError:
                    best_checkpoint = torch.load(best_path, map_location='cpu')
                self.best_val_loss = min(
                    self.best_val_loss,
                    float(best_checkpoint.get('val_loss', float('inf'))),
                )
        if self.is_main_process:
            print(
                f"Resumed training from {resume_path}: completed_epoch={completed_epoch}, "
                f"next_epoch={self.start_epoch}, best_val_loss={self.best_val_loss:.6f}"
            )

    def _print_model_info(self):
        if not self.is_main_process:
            return
        model_for_info = self.model.module if self.distributed else self.model
        num_params = sum(p.numel() for p in model_for_info.parameters())
        print("=" * 60)
        print("vEMDestripe training")
        print("=" * 60)
        print(f"数据集目录: {self.args.data_dir}")
        print("Model: vEMDestripe")
        print(f"总参数量: {num_params:,}")
        print(f"设备: {self.device} (world_size={self.world_size})")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"混合精度 AMP: {'启用' if self.use_amp else '禁用'}")
        print(f"梯度累积步数: {self.accumulation_steps} (全局有效 batch={self.args.batch_size * self.accumulation_steps * self.world_size})")
        print("=" * 60)
        print("CSG units: identity-initialized channel-spatial gates")
        print("Loss: pixel + edge + photometric moments + SSIM")
        print("=" * 60)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = {
            'total': 0, 'l1': 0, 'edge': 0, 'brightness': 0,
            'global_mean': 0, 'global_std': 0, 'local_mean': 0,
            'local_std': 0, 'ssim': 0,
        }
        num_batches = 0

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        pbar = (
            tqdm(
                self.train_loader,
                desc=f"Train Epoch {epoch}/{self.args.epochs}",
                disable=bool(getattr(self.args, 'disable_progress_bar', False)),
            )
            if self.is_main_process else self.train_loader
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(pbar):
            gt = batch['gt'].to(self.device)
            input_striped = batch['input'].to(self.device)
            reference = batch['reference'].to(self.device)
            ds_brightness = batch['ds_brightness'].to(self.device)
            ds_contrast = batch['ds_contrast'].to(self.device)

            model_input = torch.cat([input_striped, reference], dim=1)

            is_update_step = ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == total_batches)
            # DDP needs to all-reduce only when the optimizer will step.  This
            # avoids redundant gradient communication on accumulation steps.
            sync_context = self.model.no_sync() if self.distributed and not is_update_step else nullcontext()

            with sync_context:
                with self._autocast_context():
                    output = self.model(model_input)
                    loss, loss_dict = self.criterion(
                        output, gt,
                        ds_brightness=ds_brightness,
                        ds_contrast=ds_contrast
                    )

                # Keep the requested effective batch size without retaining a
                # batch-8 1024x1024 activation graph in GPU memory.  The final
                # group can be shorter than accumulation_steps.
                group_start = batch_idx - (batch_idx % self.accumulation_steps)
                group_size = min(self.accumulation_steps, total_batches - group_start)
                self.grad_scaler.scale(loss / group_size).backward()
            if is_update_step:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            epoch_losses['total'] += loss.item()
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            num_batches += 1

            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'l1': f"{loss_dict.get('l1', 0):.4f}",
                    'photo': f"{loss_dict.get('brightness', 0):.4f}",
                })

            global_step = epoch * len(self.train_loader) + batch_idx
            if self.is_main_process and batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/L1', loss_dict.get('l1', 0), global_step)
                self.writer.add_scalar('Train/Edge', loss_dict.get('edge', 0), global_step)
                self.writer.add_scalar('Train/Photometric', loss_dict.get('brightness', 0), global_step)
                self.writer.add_scalar('Train/GlobalMean', loss_dict.get('global_mean', 0), global_step)
                self.writer.add_scalar('Train/GlobalStd', loss_dict.get('global_std', 0), global_step)
                self.writer.add_scalar('Train/LocalMean', loss_dict.get('local_mean', 0), global_step)
                self.writer.add_scalar('Train/LocalStd', loss_dict.get('local_std', 0), global_step)
                self.writer.add_scalar('Train/SSIMLoss', loss_dict.get('ssim', 0), global_step)

        return self._reduce_epoch_losses(epoch_losses, num_batches)

    def validate(self, epoch):
        self.model.eval()
        val_losses = {
            'total': 0, 'l1': 0, 'edge': 0, 'brightness': 0,
            'global_mean': 0, 'global_std': 0, 'local_mean': 0,
            'local_std': 0, 'ssim': 0,
        }
        num_batches = 0

        with torch.no_grad():
            iterator = (
                tqdm(
                    self.val_loader,
                    desc="Validation",
                    disable=bool(getattr(self.args, 'disable_progress_bar', False)),
                )
                if self.is_main_process else self.val_loader
            )
            for batch in iterator:
                gt = batch['gt'].to(self.device)
                input_striped = batch['input'].to(self.device)
                reference = batch['reference'].to(self.device)
                ds_brightness = batch['ds_brightness'].to(self.device)
                ds_contrast = batch['ds_contrast'].to(self.device)

                model_input = torch.cat([input_striped, reference], dim=1)
                with self._autocast_context():
                    output = self.model(model_input)
                    loss, loss_dict = self.criterion(
                        output, gt,
                        ds_brightness=ds_brightness,
                        ds_contrast=ds_contrast
                    )

                val_losses['total'] += loss.item()
                for k, v in loss_dict.items():
                    if k in val_losses:
                        val_losses[k] += v
                num_batches += 1

        val_losses = self._reduce_epoch_losses(val_losses, num_batches)

        if self.is_main_process:
            self.writer.add_scalar('Val/Loss', val_losses['total'], epoch)
            self.writer.add_scalar('Val/L1', val_losses['l1'], epoch)
            self.writer.add_scalar('Val/Edge', val_losses['edge'], epoch)
            self.writer.add_scalar('Val/Brightness', val_losses['brightness'], epoch)
            self.writer.add_scalar('Val/GlobalMean', val_losses['global_mean'], epoch)
            self.writer.add_scalar('Val/GlobalStd', val_losses['global_std'], epoch)
            self.writer.add_scalar('Val/LocalMean', val_losses['local_mean'], epoch)
            self.writer.add_scalar('Val/LocalStd', val_losses['local_std'], epoch)
            self.writer.add_scalar('Val/SSIMLoss', val_losses['ssim'], epoch)

        return val_losses

    def save_checkpoint(self, epoch, train_loss, val_loss, best_val_loss, is_best=False):
        if not self.is_main_process:
            return
        model_for_save = self.model.module if self.distributed else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_for_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': self.args,
            'method_name': 'vEMDestripe',
            'checkpoint_schema': 1,
        }

        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到 {best_path}")

    def save_visualization(self, epoch):
        if not self.is_main_process:
            return
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            gt = batch['gt'].to(self.device)
            input_striped = batch['input'].to(self.device)
            reference = batch['reference'].to(self.device)
            datasets = batch['dataset']

            model_input = torch.cat([input_striped, reference], dim=1)
            with self._autocast_context():
                # Only rank 0 writes previews; bypass the DDP wrapper so the
                # other ranks do not need to participate in this extra pass.
                model_for_vis = self.model.module if self.distributed else self.model
                output = model_for_vis(model_input)

            for i in range(min(4, gt.size(0))):
                vis = torch.cat([
                    input_striped[i],
                    reference[i],
                    gt[i],
                    output[i]
                ], dim=2)

                vis_np = vis.cpu().numpy().squeeze()
                save_path = os.path.join(self.args.output_dir, f'epoch{epoch}_sample{i}_{datasets[i]}.png')
                cv2.imwrite(save_path, (vis_np * 255).astype(np.uint8))

    def train(self):
        if self.is_main_process:
            print("\n开始训练...")
        best_val_loss = self.best_val_loss

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate(epoch)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.is_main_process:
                print(f"\nEpoch {epoch}/{self.args.epochs}")
                print(f"  训练损失: {train_losses['total']:.4f} (L1: {train_losses.get('l1',0):.4f}, Edge: {train_losses.get('edge',0):.4f}, Photo: {train_losses.get('brightness',0):.4f}, SSIM: {train_losses.get('ssim',0):.4f})")
                print(f"  验证损失: {val_losses['total']:.4f} (L1: {val_losses.get('l1',0):.4f}, Edge: {val_losses.get('edge',0):.4f}, Photo: {val_losses.get('brightness',0):.4f}, SSIM: {val_losses.get('ssim',0):.4f})")
                print(f"  学习率: {current_lr:.6f}")

                is_best = val_losses['total'] < best_val_loss
                if is_best:
                    best_val_loss = val_losses['total']
                self.save_checkpoint(
                    epoch,
                    train_losses['total'],
                    val_losses['total'],
                    best_val_loss,
                    is_best,
                )

                if epoch % self.args.save_freq == 0:
                    self.save_visualization(epoch)
            self._barrier()

        if self.is_main_process:
            print("\n训练完成!")
            self.writer.close()
        if self.distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Train vEMDestripe on paired simulated vEM sections.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data_dir', type=str, required=True,
                        help="数据集目录")
    parser.add_argument('--image_size', type=int, default=1024,
                        help="图像处理尺寸 (默认: 1024)")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="每个 GPU 的 batch size")
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help="累积多少个物理 batch 后再更新参数（用于在显存受限时保持有效 batch）")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument('--amp', action='store_true',
                        help="在 CUDA 上使用 FP16 自动混合精度以加快训练并降低显存占用")

    parser.add_argument('--disable_progress_bar', action='store_true',
                        help='Disable tqdm output in Slurm logs; epoch summaries are still printed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume model, optimizer, scheduler and AMP state from a checkpoint')

    parser.add_argument('--epochs', type=int, default=200,
                        help="训练轮数")
    parser.add_argument('--lr', type=float, default=2e-4,
                        help="学习率")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU编号")
    parser.add_argument('--save_freq', type=int, default=10,
                        help="可视化保存频率")
    parser.add_argument('--seed', type=int, default=42,
                        help="随机种子")

    # Fixed objective reported for vEMDestripe.
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                        help="L1重建损失权重")
    parser.add_argument('--lambda_edge', type=float, default=0.1,
                        help="边缘损失权重")
    # Reference-guided photometric loss settings.
    parser.add_argument('--lambda_photo', type=float, default=0.3,
                        help='Global/local photometric-statistics loss weight')
    parser.add_argument('--lambda_ssim', type=float, default=0.1,
                        help='SSIM loss weight')
    parser.add_argument('--photo_std_weight', type=float, default=0.5,
                        help='Relative std weight in global/local photometric terms')
    parser.add_argument('--local_window_sizes', type=str, default='32,64,128',
                        help='Comma-separated local photometric pooling windows')

    parser.add_argument('--use_fused_reference', action='store_true',
                        help="使用融合参考图像")
    parser.add_argument('--no_fused_reference', action='store_true',
                        help="不使用融合参考图像")
    parser.add_argument('--reference_mode', type=str, default='fused', choices=['fused', 'single'],
                        help="参考图输入：fused=多张融合后结构破坏图；single=一张单参考图的结构破坏图")
    parser.add_argument('--reference_block_size', type=int, default=16,
                        help="block_mean 参考图预处理的非重叠分块大小")
    parser.add_argument(
        '--reference_preprocess', type=str, default='block_mean',
        choices=['stored', 'raw_mean', 'block_mean', 'zero', 'self'],
        help=(
            '训练参考图：stored=读取数据生成器保存的融合参考（旧行为）；'
            'raw_mean=原始单参考均值；block_mean=原始单参考均值后分块均值；'
            'zero=全零第二通道；self=输入图自身作为第二通道'
        ),
    )
    parser.add_argument('--single_reference_policy', type=str, default='random', choices=['random', 'fixed'],
                        help="single 模式中每个样本选择单参考图的策略")
    parser.add_argument('--single_reference_index', type=int, default=0,
                        help="single_reference_policy=fixed 时使用的参考图索引")

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/vemdestripe',
                        help="检查点保存目录")
    parser.add_argument('--output_dir', type=str, default='./outputs/vemdestripe',
                        help="可视化结果保存目录")
    parser.add_argument('--log_dir', type=str, default='./logs/tensorboard_vemdestripe',
                        help="TensorBoard日志目录")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    args.method_name = 'vEMDestripe'
    args.checkpoint_schema = 1
    trainer = VEMDestripeTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
