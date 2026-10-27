#!/usr/bin/env python3
"""Run vEMDestripe inference on a TIFF stack or an image directory.

Exactly 1024 x 1024 sections bypass tiled blending. Larger or differently
sized sections use reflection-padded overlapping windows whose exterior edges
retain unit weight, preventing the one-pixel black border found in the legacy
cosine blender.
"""

import os
import sys
import argparse
import json
import hashlib
from pathlib import Path
import shutil
import numpy as np
import torch
import cv2
import tifffile as tiff
from tqdm import tqdm
from glob import glob
try:
    from natsort import natsorted
except ImportError:
    # Keep local CPU inference usable in minimal environments.  The expected
    # slice filenames are zero-padded (for example, train_002133.png), so the
    # built-in lexical ordering preserves their numerical order.
    def natsorted(items):
        return sorted(items)

from vemdestripe.model import VEMDestripe
from vemdestripe.reference import block_average


def imread_grayscale(path):
    """Read a grayscale image while supporting Windows paths with Unicode."""
    try:
        raw = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def imwrite_image(path, image):
    """Write an image while supporting Windows paths with Unicode."""
    suffix = os.path.splitext(os.fspath(path))[1] or '.png'
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(os.fspath(path))
    return True


def destroy_reference_structure(image):
    """Destroy all spatial structure while preserving the exact intensity histogram.

    This matches the historical single-reference compatibility path: a deterministic
    full-image pixel permutation preserves brightness, contrast and histogram,
    but retains no anatomical or texture position information.
    """
    if image is None:
        return None
    source = np.ascontiguousarray(image)
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
    """Match V3 training: replace every block by its mean, then quantize."""
    if image is None:
        return None
    if block_size < 1:
        raise ValueError('block_size must be positive')
    result = block_average(image, block_size)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def load_z_sequence(image_paths, current_idx, window_size=8, reference_image=None):
    """
    加载Z轴序列（相邻切片）
    
    Args:
        image_paths: 所有图像路径列表
        current_idx: 当前处理的图像索引
        window_size: Z轴窗口大小（奇数）
        reference_image: 可选的参考图像
    
    Returns:
        z_sequence: torch.Tensor, shape [Z, 2, H, W]
                   Z = window_size，2通道：[当前/相邻切片, reference]
    """
    Z = window_size
    half_window = Z // 2
    
    # 计算窗口边界
    start_idx = max(0, current_idx - half_window)
    end_idx = min(len(image_paths), current_idx + half_window + 1)
    
    # 调整窗口以确保正好有Z个切片
    actual_size = end_idx - start_idx
    if actual_size < Z:
        if start_idx == 0:
            end_idx = min(len(image_paths), start_idx + Z)
        else:
            start_idx = max(0, end_idx - Z)
    
    # 加载窗口内的图像
    z_images = []
    for idx in range(start_idx, end_idx):
        img = imread_grayscale(image_paths[idx])
        if img is None:
            if len(z_images) > 0:
                img = z_images[-1].copy()
            else:
                img = np.zeros((256, 256), dtype=np.uint8)
        img_float = img.astype(np.float32) / 255.0
        
        # 与参考图像拼接
        if reference_image is not None:
            ref_float = reference_image.astype(np.float32) / 255.0
            combined = np.stack([img_float, ref_float], axis=0)
        else:
            combined = np.stack([img_float, img_float], axis=0)
        
        z_images.append(combined)
    
    z_sequence = np.stack(z_images, axis=0)
    return torch.from_numpy(z_sequence).float()


class ZAxisPostProcess:
    """Z轴后处理"""

    def __init__(self, smooth_kernel=3, outlier_threshold=3.0):
        self.smooth_kernel = smooth_kernel
        self.outlier_threshold = outlier_threshold

    def detect_outliers(self, volume):
        Z, H, W = volume.shape
        brightness_means = np.array([np.mean(volume[z]) for z in range(Z)])

        global_mean = np.median(brightness_means)
        global_std = np.std(brightness_means)

        outliers = []
        for z in range(Z):
            if abs(brightness_means[z] - global_mean) > self.outlier_threshold * global_std:
                outliers.append(z)

        return np.array(outliers)

    def smooth_sequence(self, volume, outlier_indices):
        Z, H, W = volume.shape
        result = volume.copy().astype(np.float32)

        for z in outlier_indices:
            neighbors = []
            for offset in [-2, -1, 1, 2]:
                nz = z + offset
                if 0 <= nz < Z and nz not in outlier_indices:
                    neighbors.append(volume[nz])

            if len(neighbors) > 0:
                target_mean = np.median([np.mean(n) for n in neighbors])
                current_mean = np.mean(volume[z])
                result[z] = volume[z] + (target_mean - current_mean)
            else:
                global_mean = np.median([np.mean(volume[i]) for i in range(Z)])
                current_mean = np.mean(volume[z])
                result[z] = volume[z] + (global_mean - current_mean)

        return result.astype(np.uint8)

    def process(self, volume):
        if volume.ndim != 3:
            return volume

        Z, H, W = volume.shape
        outliers = self.detect_outliers(volume)

        if len(outliers) > 0:
            print(f"  [optional Z postprocess] detected {len(outliers)} outlier sections")
            volume_smoothed = self.smooth_sequence(volume, outliers.tolist())

            if self.smooth_kernel > 1:
                kernel = np.ones((self.smooth_kernel, 1, 1), dtype=np.float32) / self.smooth_kernel
                padded = np.pad(volume_smoothed, ((self.smooth_kernel//2, self.smooth_kernel//2), (0, 0), (0, 0)), mode='edge')
                for z in range(Z):
                    volume_smoothed[z] = np.mean(padded[z:z+self.smooth_kernel], axis=0)

            result = volume_smoothed
        else:
            result = volume

        return result.astype(np.uint8)


class VEMDestripeInferencer:
    """Inference engine for the released vEMDestripe architecture."""

    def __init__(self, model_path, device='cuda', gpu=0,
                 image_size=1024, inference_size=1024, overlap_ratio=0.25,
                 use_sliding_window=True, use_z_postprocess=False,
                 upsample_xy=1.0, upsample_order=1,
                 destroy_reference_struct=True, struct_block_size=16, struct_num_keep_blocks=0,
                 z_sequence_length=8, use_z_context=False,
                 reference_mode='fused', single_reference_index=0,
                 legacy_per_image_normalization=False,
                 train_compatible_reference=True, disable_progress_bar=False,
                 reference_preprocess='legacy', reference_selection='legacy',
                 reference_seed=42):
        """
        Args:
            z_sequence_length: compatibility field; the released model does not use Z context.
            use_z_context: compatibility field; ignored by the released model.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.inference_size = inference_size
        self.overlap_ratio = overlap_ratio
        # 【修复】滑动窗口逻辑：当推理窗口小于图像尺寸时启用滑动窗口
        # Keep the user's preference. process_single checks the actual image
        # shape, so a 1024x1024 image with a 1024 window uses a full-image pass.
        self.use_sliding_window = bool(use_sliding_window)
        self.use_z_postprocess = use_z_postprocess
        self.upsample_xy = upsample_xy
        self.upsample_order = upsample_order

        # 【新增】Reference结构破坏参数
        self.destroy_reference_struct = destroy_reference_struct
        self.struct_block_size = struct_block_size
        self.struct_num_keep_blocks = struct_num_keep_blocks
        if reference_mode not in ('fused', 'single'):
            raise ValueError(f'Unknown reference_mode: {reference_mode}')
        self.reference_mode = reference_mode
        self.single_reference_index = single_reference_index
        self.legacy_per_image_normalization = legacy_per_image_normalization
        self.train_compatible_reference = bool(train_compatible_reference)
        valid_reference_preprocess = ('legacy', 'block_mean', 'raw_mean', 'zero', 'self')
        if reference_preprocess not in valid_reference_preprocess:
            raise ValueError(
                f'Unknown reference_preprocess={reference_preprocess}; '
                f'expected one of {valid_reference_preprocess}'
            )
        if reference_selection not in ('legacy', 'uniform', 'random'):
            raise ValueError('reference_selection must be legacy, uniform, or random')
        self.reference_preprocess = reference_preprocess
        self.reference_selection = reference_selection
        self.reference_seed = int(reference_seed)
        self.disable_progress_bar = bool(disable_progress_bar)

        # The released network processes XY sections independently.
        self.z_sequence_length = z_sequence_length
        self.use_z_context = False

        self.model = VEMDestripe(
            in_channels=2,
            out_channels=1,
            base_features=32,
            num_blocks=2,
        )
        print("Model: vEMDestripe")
        self.model = self.model.to(self.device)

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError('Checkpoint does not contain a model state dictionary')
        state_dict = {
            str(key).removeprefix('module.'): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        print(f"Checkpoint: {model_path}")
        print(f"设备: {self.device}")
        print(f"训练尺寸: {image_size}, 推理尺寸: {inference_size}")
        if self.reference_preprocess != 'legacy':
            reference_description = self.reference_preprocess
            if self.reference_preprocess == 'block_mean':
                reference_description += f' ({self.struct_block_size}x{self.struct_block_size})'
        elif self.train_compatible_reference:
            reference_description = f'mean fusion + {self.struct_block_size}x{self.struct_block_size} block mean'
        else:
            reference_description = 'global pixel permutation' if destroy_reference_struct else 'disabled'
        print(f"Reference mode: {reference_mode}; preprocessing: {reference_description}")
        print("Z context: disabled (reference-guided 2-D inference)")

        if self.use_z_postprocess:
            self.z_postprocess = ZAxisPostProcess()

    def upsample_xy_func(self, image):
        if self.upsample_xy == 1.0:
            return image

        h, w = image.shape[:2]
        new_h, new_w = int(h * self.upsample_xy), int(w * self.upsample_xy)

        if self.upsample_order == 0:
            method = cv2.INTER_NEAREST
        elif self.upsample_order == 1:
            method = cv2.INTER_LINEAR
        elif self.upsample_order == 3:
            method = cv2.INTER_CUBIC
        else:
            method = cv2.INTER_LINEAR

        return cv2.resize(image, (new_w, new_h), interpolation=method)

    def downsample_xy_func(self, image, orig_size):
        if self.upsample_xy == 1.0:
            return image
        return cv2.resize(image, (orig_size[1], orig_size[0]))

    def create_reference_image(self, image_paths, num_samples=50):
        """Create a fused or a single reference using the training-compatible rule."""
        if not image_paths:
            raise ValueError('No images are available to create a reference')
        if self.reference_mode == 'single':
            index = self.single_reference_index % len(image_paths)
            reference = imread_grayscale(image_paths[index])
            if reference is None:
                raise ValueError(f'Unable to read single reference: {image_paths[index]}')
            return self._destroy_reference(reference), [index]

        indices = self._select_reference_indices(len(image_paths), num_samples, source='automatic')
        images = [imread_grayscale(image_paths[index]) for index in indices]
        images = [image.astype(np.float32) for image in images if image is not None]
        if not images:
            raise ValueError('Unable to read any reference image')
        return self._fuse_reference_images(images), indices

    def _select_reference_indices(self, length, num_reference, source='automatic'):
        count = min(int(num_reference), int(length))
        if count < 1:
            raise ValueError('num_reference must be positive')
        if count == length:
            return list(range(length))
        selection = self.reference_selection
        if selection == 'legacy':
            selection = 'random' if source == 'automatic' else 'uniform'
        if selection == 'random':
            rng = np.random.default_rng(self.reference_seed)
            return sorted(rng.choice(length, count, replace=False).tolist())
        return np.linspace(0, length - 1, count, dtype=int).tolist()

    def _fuse_reference_images(self, images):
        if not images:
            raise ValueError('Unable to fuse an empty reference list')
        mode = self.reference_preprocess
        if mode == 'self':
            return None
        if mode == 'zero':
            return np.zeros_like(images[0], dtype=np.uint8)
        if mode in ('block_mean', 'raw_mean'):
            reference = np.mean(images, axis=0, dtype=np.float32)
            if mode == 'block_mean':
                return block_mean_reference_structure(reference, self.struct_block_size)
            return np.clip(reference, 0.0, 255.0).astype(np.uint8)
        if self.train_compatible_reference:
            reference = np.mean(images, axis=0, dtype=np.float32)
        else:
            reference = np.median(images, axis=0).astype(np.uint8)
        return self._destroy_reference(reference)

    def _destroy_reference(self, reference):
        if self.reference_preprocess == 'self':
            return None
        if self.reference_preprocess == 'zero':
            return np.zeros_like(reference, dtype=np.uint8)
        if self.reference_preprocess == 'block_mean':
            return block_mean_reference_structure(reference, self.struct_block_size)
        if self.reference_preprocess == 'raw_mean':
            return np.clip(reference, 0.0, 255.0).astype(np.uint8)
        if self.train_compatible_reference:
            return block_mean_reference_structure(reference, self.struct_block_size)
        return destroy_reference_structure(reference) if self.destroy_reference_struct else reference

    def load_external_reference(self, reference_path, num_reference):
        """Load one reference or fuse multiple references before structure destruction."""
        if not os.path.isdir(reference_path):
            reference = imread_grayscale(reference_path)
            if reference is None:
                raise ValueError(f'Unable to read reference image: {reference_path}')
            return self._destroy_reference(reference), [os.path.basename(reference_path)]

        reference_files = []
        for extension in ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg'):
            reference_files.extend(glob(os.path.join(reference_path, extension)))
            reference_files.extend(glob(os.path.join(reference_path, extension.upper())))
        reference_files = natsorted(list(dict.fromkeys(reference_files)))
        if not reference_files:
            raise ValueError(f'No reference images found in: {reference_path}')

        if self.reference_mode == 'single':
            path = reference_files[self.single_reference_index % len(reference_files)]
            reference = imread_grayscale(path)
            if reference is None:
                raise ValueError(f'Unable to read single reference: {path}')
            return self._destroy_reference(reference), [os.path.basename(path)]

        indices = self._select_reference_indices(len(reference_files), num_reference, source='external')
        selected_paths = [reference_files[index] for index in indices]
        images = [imread_grayscale(path) for path in selected_paths]
        images = [image.astype(np.float32) for image in images if image is not None]
        if not images:
            raise ValueError(f'Unable to read references from: {reference_path}')
        return self._fuse_reference_images(images), [os.path.basename(path) for path in selected_paths]

    def preprocess(self, image, reference):
        orig_size = image.shape[:2]

        if self.upsample_xy != 1.0:
            image = self.upsample_xy_func(image)
            reference = self.upsample_xy_func(reference)

        image_p = image.astype(np.float32) / 255.0
        ref_p = reference.astype(np.float32) / 255.0

        return image_p, ref_p, orig_size

    def _sliding_window_inference(self, image, reference):
        h, w = image.shape[:2]
        # 【修复】使用 inference_size 而不是 image_size 作为滑动窗口大小
        window_size = self.inference_size
        overlap_ratio = self.overlap_ratio
        stride = max(1, int(window_size * (1 - overlap_ratio)))

        pad_h = max(0, window_size - h)
        pad_w = max(0, window_size - w)

        image_pad = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        ref_pad = np.pad(reference, ((0, pad_h), (0, pad_w)), mode='reflect')

        output = np.zeros_like(image_pad, dtype=np.float64)
        weight = np.zeros_like(image_pad, dtype=np.float64)

        # 生成所有滑动窗口位置
        positions = []
        for y in range(0, image_pad.shape[0] - window_size + 1, stride):
            for x in range(0, image_pad.shape[1] - window_size + 1, stride):
                positions.append((y, x))

        # 确保边角被覆盖
        edge_positions = [
            (0, 0),
            (0, max(0, image_pad.shape[1] - window_size)),
            (max(0, image_pad.shape[0] - window_size), 0),
            (max(0, image_pad.shape[0] - window_size),
             max(0, image_pad.shape[1] - window_size))
        ]
        for pos in edge_positions:
            if pos not in positions:
                positions.append(pos)

        positions = sorted(list(set(positions)))

        # 计算淡入淡出区域大小
        fade = max(1, int(window_size * overlap_ratio)) if overlap_ratio > 0 else 0

        for y, x in tqdm(
            positions,
            desc="滑动窗口",
            leave=False,
            disable=self.disable_progress_bar,
        ):
            window = image_pad[y:y+window_size, x:x+window_size]
            ref_window = ref_pad[y:y+window_size, x:x+window_size]

            window_t = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).float().to(self.device)
            ref_t = torch.from_numpy(ref_window).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                model_input = torch.cat([window_t, ref_t], dim=1)
                output_t = self.model(model_input)

            patch_result = output_t.squeeze().cpu().numpy()

            # 创建2D淡入淡出权重（使用余弦函数确保平滑过渡）
            patch_weight = np.ones((window_size, window_size), dtype=np.float64)

            if fade > 0 and overlap_ratio > 0:
                # 使用余弦插值获得更平滑的权重
                fade_weights = (1 - np.cos(np.linspace(0, np.pi, fade))) / 2

                # 四边分别应用权重
                fade_weights = (1 - np.cos(np.linspace(0, np.pi, fade))) / 2
                weight_y = np.ones(window_size, dtype=np.float64)
                weight_x = np.ones(window_size, dtype=np.float64)

                # Only taper edges that overlap another patch. Keeping unit
                # weight at the outside boundary prevents a black border.
                if y > 0:
                    weight_y[:fade] = fade_weights
                if y + window_size < image_pad.shape[0]:
                    weight_y[-fade:] = fade_weights[::-1]
                if x > 0:
                    weight_x[:fade] = fade_weights
                if x + window_size < image_pad.shape[1]:
                    weight_x[-fade:] = fade_weights[::-1]
                patch_weight = np.outer(weight_y, weight_x)

            # 累加到输出
            output[y:y+window_size, x:x+window_size] += patch_result.astype(np.float64) * patch_weight
            weight[y:y+window_size, x:x+window_size] += patch_weight

        # 归一化处理
        weight[weight < 1e-6] = 1.0  # 防止除零
        output = output / weight

        # 裁剪到原始尺寸
        output = output[:h, :w]

        # 确保输出在有效范围内
        output = np.clip(output, 0, 1)

        return output

    def process_single(self, image, reference, use_sliding_window=None, z_sequence=None):
        image_p, ref_p, orig_size = self.preprocess(image, reference)

        if use_sliding_window is None:
            use_sliding_window = self.use_sliding_window
        # Exact window-sized inputs need no blending. Inputs of any other size
        # retain the existing window path (tiling when larger, padding when
        # smaller), which also keeps the model input size predictable.
        requires_windowing = image_p.shape[:2] != (
            self.inference_size,
            self.inference_size,
        )
        use_sliding_window = bool(use_sliding_window and requires_windowing)

        if use_sliding_window:
            result = self._sliding_window_inference(image_p, ref_p)
        else:
            image_t = torch.from_numpy(image_p).unsqueeze(0).unsqueeze(0).float().to(self.device)
            ref_t = torch.from_numpy(ref_p).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                model_input = torch.cat([image_t, ref_t], dim=1)
                # vEMDestripe does not consume an explicit Z sequence.
                if z_sequence is not None:
                    print("  Warning: explicit Z context is not part of vEMDestripe and was ignored")
                output = self.model(model_input)

            result = output.squeeze().cpu().numpy()

        if self.upsample_xy != 1.0:
            result = self.downsample_xy_func(result, orig_size)

        result = self._postprocess(result)

        return result

    def _postprocess(self, result):
        """Convert model output to uint8 without erasing predicted brightness.

        V4 brightness-domain training asks the network to predict an absolute
        target intensity. The historical per-image percentile normalization is
        only available as an opt-in compatibility mode because it removes that
        learned absolute-brightness information.
        """
        if not self.legacy_per_image_normalization:
            return np.rint(np.clip(result * 255.0, 0.0, 255.0)).astype(np.uint8)

        result_norm = (result - result.min()) / (result.max() - result.min() + 1e-6)
        p_low, p_high = np.percentile(result_norm, (1, 99))
        result_norm = np.clip(result_norm, p_low, p_high)
        result_norm = (result_norm - result_norm.min()) / (result_norm.max() - result_norm.min() + 1e-6)

        result = result_norm * 255.0
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def process_directory(self, input_dir, output_dir, reference=None, num_reference=50,
                          save_preview=True, use_sliding_window=True,
                          output_format='png',
                          use_z_context=None, z_sequence_length=None,
                          no_reference=False, max_images=None):
        """
        处理目录中的图像
        
        Args:
            use_z_context: 是否使用Z轴上下文（None则使用初始化时的值）
            z_sequence_length: Z轴窗口大小（None则使用初始化时的值）
        """
        if use_z_context is None:
            use_z_context = self.use_z_context
        if z_sequence_length is None:
            z_sequence_length = self.z_sequence_length
        
        os.makedirs(output_dir, exist_ok=True)

        image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(input_dir, ext)))
            image_paths.extend(glob(os.path.join(input_dir, ext.upper())))

        # Windows glob is case-insensitive, so '*.png' and '*.PNG' can match
        # the same file. Deduplicate before sorting to avoid duplicate output.
        unique_paths = {}
        for path in image_paths:
            unique_paths.setdefault(os.path.normcase(os.path.abspath(path)), path)
        image_paths = natsorted(list(unique_paths.values()))

        total_input_images = len(image_paths)
        if max_images is not None and max_images < total_input_images:
            if max_images <= 0:
                raise ValueError("max_images must be a positive integer")
            selected = np.linspace(0, total_input_images - 1, max_images, dtype=int)
            image_paths = [image_paths[i] for i in selected]
            print(f"Uniformly selected {len(image_paths)} of {total_input_images} input images")

        if len(image_paths) == 0:
            raise ValueError(f"在 {input_dir} 中没有找到图像文件")

        print(f"找到 {len(image_paths)} 张图像")

        # 【消融实验】无参考模式
        if no_reference or self.reference_preprocess == 'self':
            print("\n【消融实验】self 模式: 使用输入图像自身作为参考")
            # 创建零图像作为占位符，实际推理时使用输入图像
            reference_image = None
            selected_indices = None
        elif reference is None:
            print(f"\n自动生成参考图像 (使用 {num_reference} 张切片)...")
            reference_image, selected_indices = self.create_reference_image(
                image_paths, num_samples=num_reference
            )
        else:
            print(f"\n加载参考图像: {reference}")
            reference_image, selected_indices = self.load_external_reference(reference, num_reference)

        # 保存参考图像信息（无参考模式跳过）
        if reference_image is not None:
            ref_path = os.path.join(output_dir, 'reference.png')
            imwrite_image(ref_path, reference_image)
        else:
            ref_path = None
            print("[消融实验] 无参考模式 - 不保存参考图像")

        info = {
            'reference_path': ref_path,
            'num_images': len(image_paths),
            'num_input_images_total': total_input_images,
            'selection': self.reference_selection,
            'num_reference_used': 0 if reference_image is None else len(selected_indices or []),
            'reference_indices': selected_indices,
            'z_postprocess': self.use_z_postprocess,
            'output_format': output_format,
            'method_name': 'vEMDestripe',
            'checkpoint_schema': 1,
            'reference_mode': self.reference_mode,
            'reference_preprocess': self.reference_preprocess,
            'reference_selection': self.reference_selection,
            'reference_seed': self.reference_seed,
            'destroy_reference_struct': self.destroy_reference_struct,
            'train_compatible_reference': self.train_compatible_reference,
            'reference_structure_destruction': (
                f'mean_fusion_then_block_mean_{self.struct_block_size}'
                if self.reference_preprocess == 'block_mean' else
                ('mean_fusion_no_structure_destruction'
                 if self.reference_preprocess == 'raw_mean' else
                 ('zero_second_channel'
                  if self.reference_preprocess == 'zero' else
                  ('self_second_channel'
                   if self.reference_preprocess == 'self' else
                   (f'mean_fusion_then_block_mean_{self.struct_block_size}'
                    if self.train_compatible_reference else
                    ('global_pixel_permutation' if self.destroy_reference_struct else 'disabled')))))
            ),
            'use_z_context': use_z_context,
            'z_sequence_length': z_sequence_length,
            'no_reference_mode': reference_image is None  # 标记无参考模式
        }

        print(f"\n开始处理 {len(image_paths)} 张图像...")
        print(f"滑动窗口: {'启用' if use_sliding_window else '禁用'}")
        print(f"输出格式: {output_format.upper()}")
        print(f"Z轴后处理: {'启用' if self.use_z_postprocess else '禁用'}")
        
        # Explicit Z sequences are outside the released method.
        if use_z_context:
            print(f"Z轴上下文: 已请求 (window={z_sequence_length})")
            print("  Warning: explicit Z context was requested but is not used")
        else:
            print(f"Z轴上下文: 禁用")

        # 【新增】预处理所有图像以获取尺寸
        first_img = imread_grayscale(image_paths[0])
        if first_img is None:
            raise ValueError(f"无法读取图像: {image_paths[0]}")
        img_height, img_width = first_img.shape
        
        # 【新增】预加载所有图像到内存（用于Z轴序列）
        if use_z_context and len(image_paths) > 1:
            print(f"\n预加载所有图像用于Z轴上下文 (共 {len(image_paths)} 张)...")
            all_images = []
            for path in tqdm(
                image_paths,
                desc="预加载",
                disable=self.disable_progress_bar,
            ):
                img = imread_grayscale(path)
                if img is None:
                    img = np.zeros((img_height, img_width), dtype=np.uint8)
                all_images.append(img)
            all_images = np.stack(all_images) if len(image_paths) > 1 else np.array([all_images[0]])
        else:
            all_images = None

        def get_z_sequence(current_idx):
            """获取当前索引的Z轴序列"""
            if not use_z_context or all_images is None:
                return None
            
            Z = z_sequence_length
            half_window = Z // 2
            
            start_idx = max(0, current_idx - half_window)
            end_idx = min(len(image_paths), current_idx + half_window + 1)
            
            actual_size = end_idx - start_idx
            if actual_size < Z:
                if start_idx == 0:
                    end_idx = min(len(image_paths), start_idx + Z)
                else:
                    start_idx = max(0, end_idx - Z)
            
            z_images = all_images[start_idx:end_idx]
            
            if len(z_images) < Z:
                pad_count = Z - len(z_images)
                if start_idx == 0:
                    z_images = np.concatenate([z_images, np.tile(z_images[-1:], (pad_count, 1, 1))], axis=0)
                else:
                    z_images = np.concatenate([np.tile(z_images[:1], (pad_count, 1, 1)), z_images], axis=0)
            
            z_tensors = []
            for z_idx, z_img in enumerate(z_images):
                img_float = z_img.astype(np.float32) / 255.0
                # 【修复】无参考模式：使用输入图像自身作为参考
                if reference_image is None:
                    ref_float = z_img.astype(np.float32) / 255.0
                else:
                    ref_float = reference_image.astype(np.float32) / 255.0
                combined = np.stack([img_float, ref_float], axis=0)
                z_tensors.append(torch.from_numpy(combined).float())
            
            z_sequence = torch.stack(z_tensors, dim=0).to(self.device)
            return z_sequence

        if self.use_z_postprocess and len(image_paths) > 1:
            print("\n使用Z轴后处理...")
            results = []

            for i, image_path in enumerate(tqdm(
                image_paths,
                desc="处理",
                disable=self.disable_progress_bar,
            )):
                image = imread_grayscale(image_path)
                if image is None:
                    results.append(None)
                    continue

                z_seq = get_z_sequence(i)

                # 【消融实验】无参考模式：使用输入图像自身作为参考
                if reference_image is None:
                    ref_for_this_image = image.copy()
                else:
                    ref_for_this_image = reference_image

                result = self.process_single(image, ref_for_this_image,
                                           use_sliding_window=use_sliding_window,
                                           z_sequence=z_seq)
                results.append(result)

            valid_results = [r for r in results if r is not None]
            if valid_results:
                volume = np.stack(valid_results)
                volume = self.z_postprocess.process(volume)
                processed_results = [volume[i] for i in range(len(valid_results))]

                idx = 0
                for i, r in enumerate(results):
                    if r is not None:
                        results[i] = processed_results[idx]
                        idx += 1

            for i, (image_path, result) in enumerate(zip(image_paths, results)):
                if result is None:
                    continue

                basename = os.path.basename(image_path)
                name, ext = os.path.splitext(basename)

                if output_format in ['png', 'both']:
                    png_path = os.path.join(output_dir, f"{name}_deflickered.png")
                    imwrite_image(png_path, result)

                if output_format in ['tif', 'both']:
                    tif_path = os.path.join(output_dir, f"{name}_deflickered.tif")
                    tiff.imwrite(tif_path, result.astype(np.uint8))
        else:
            for i, image_path in enumerate(tqdm(
                image_paths,
                desc="处理",
                disable=self.disable_progress_bar,
            )):
                image = imread_grayscale(image_path)
                if image is None:
                    continue

                z_seq = get_z_sequence(i)

                # 【消融实验】无参考模式：使用输入图像自身作为参考
                if reference_image is None:
                    ref_for_this_image = image.copy()
                else:
                    ref_for_this_image = reference_image

                result = self.process_single(image, ref_for_this_image,
                                           use_sliding_window=use_sliding_window,
                                           z_sequence=z_seq)

                basename = os.path.basename(image_path)
                name, ext = os.path.splitext(basename)

                if output_format in ['png', 'both']:
                    png_path = os.path.join(output_dir, f"{name}_deflickered.png")
                    imwrite_image(png_path, result)

                if output_format in ['tif', 'both']:
                    tif_path = os.path.join(output_dir, f"{name}_deflickered.tif")
                    tiff.imwrite(tif_path, result.astype(np.uint8))

        print(f"\n处理完成! 结果保存在: {output_dir}")

        if save_preview and len(image_paths) > 0:
            preview_dir = os.path.join(output_dir, 'previews')
            os.makedirs(preview_dir, exist_ok=True)
            self._save_previews(image_paths, output_dir, reference_image, preview_dir)

        info_path = os.path.join(output_dir, 'processing_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        return reference_image

    def _save_previews(self, image_paths, output_dir, reference_image, preview_dir, max_images=8):
        print(f"\n生成预览图...")

        num_images = min(max_images, len(image_paths))
        indices = np.linspace(0, len(image_paths) - 1, num_images, dtype=int)

        for idx in indices:
            image_path = image_paths[idx]
            basename = os.path.basename(image_path)
            name, _ = os.path.splitext(basename)

            original = imread_grayscale(image_path)
            if original is None:
                continue

            processed = self.process_single(original, reference_image)

            h = original.shape[0]
            preview = np.zeros((h * 3, original.shape[1]), dtype=np.uint8)
            preview[0:h, :] = original
            preview[h:2*h, :] = reference_image
            preview[2*h:3*h, :] = processed

            preview_color = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            cv2.putText(preview_color, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(preview_color, "Reference", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(preview_color, "Processed", (10, 2*h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            save_path = os.path.join(preview_dir, f"{name}_preview.png")
            imwrite_image(save_path, preview_color)

        print(f"预览图保存在: {preview_dir}")


def run_directory_pipeline(inferencer, args):
    """Network -> all-volume reference calibration -> saved final outputs.

    Evaluation is a separate read-only stage. Network uint8 files are kept in
    network_raw/; only final files are written to the requested output root.
    """
    final_root = Path(args.output_dir)
    calibrate = not args.no_photometric_calibration
    if calibrate and (args.no_reference or inferencer.reference_preprocess == 'self'):
        raise ValueError('Reference calibration requires original references; use --no_photometric_calibration for a raw ablation')
    if calibrate and args.max_images is not None:
        raise ValueError('Whole-volume calibration cannot use --max_images. Infer every slice or explicitly request --no_photometric_calibration')
    raw_root = final_root / 'network_raw' if calibrate else final_root
    if calibrate and (raw_root.exists() or (final_root/'CALIBRATION.json').exists()
                      or any(final_root.glob('*_deflickered.*'))):
        raise FileExistsError('Choose a new output directory; existing network/final results are not overwritten')
    inferencer.process_directory(
        input_dir=args.input_dir, output_dir=str(raw_root), reference=args.reference,
        num_reference=args.num_reference, save_preview=False if calibrate else not args.no_preview,
        use_sliding_window=not args.no_sliding_window, output_format=args.output_format,
        use_z_context=bool(args.use_z_context and not args.no_z_context),
        z_sequence_length=args.z_sequence_length, no_reference=args.no_reference,
        max_images=args.max_images)
    if not calibrate:
        return
    from vemdestripe.calibration import calibrate_directory
    info = json.loads((raw_root/'processing_info.json').read_text(encoding='utf-8'))
    selected = info['reference_indices']
    if args.reference:
        ref_root = Path(args.reference)
        original_refs = [ref_root/name for name in selected] if ref_root.is_dir() else [ref_root]
    else:
        patterns = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
        inputs = natsorted([str(p) for p in Path(args.input_dir).iterdir()
                            if p.is_file() and p.suffix.lower() in patterns])
        original_refs = [Path(inputs[int(index)]) for index in selected]
    primary = '*_deflickered.tif' if args.output_format == 'tif' else '*_deflickered.png'
    patterns = ('*_deflickered.png', '*_deflickered.tif') if args.output_format == 'both' else (primary,)
    record = calibrate_directory(raw_root, final_root, original_refs,
        input_pattern=primary, write_patterns=patterns, expected_count=info['num_input_images_total'])
    info.update(output_pipeline='network_then_whole_volume_reference_calibration',
                photometric_calibration_applied=True, network_raw_directory='network_raw',
                calibration_record='CALIBRATION.json', calibration_source_sections=record['source_statistics']['section_count'],
                calibration_target='pooled selected original reference snapshots, not reference.png',
                final_output_note='Evaluate the saved *_deflickered files in this directory; never normalize again')
    (final_root/'processing_info.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if (raw_root/'reference.png').is_file():
        shutil.copy2(raw_root/'reference.png',final_root/'reference.png')
    print('vEMDestripe complete: network_raw retained; reference-calibrated final images saved to',final_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run vEMDestripe on an image directory or multi-page TIFF. "
            "The reference is a float mean followed by non-overlapping block averaging."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', '-i', type=str, default=None,
                        help="输入TIF文件")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="输出TIF文件")
    parser.add_argument('--input_dir', type=str, default=None,
                        help="输入目录")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="输出目录")
    parser.add_argument('--model', '-m', type=str, required=True,
                        help="模型权重路径")

    parser.add_argument('--reference', '-r', type=str, default=None,
                        help="参考图像路径")
    parser.add_argument('--image_size', type=int, default=1024,
                        help="训练时的图像尺寸")
    parser.add_argument('--inference_size', type=int, default=1024,
                        help="推理时的图像尺寸")
    parser.add_argument('--overlap_ratio', type=float, default=0.25,
                        help="滑动窗口重叠比例")
    parser.add_argument('--no_sliding_window', action='store_true',
                        help="禁用滑动窗口")
    parser.add_argument('--num_reference', type=int, default=16,
                        help="参考切片数量；16 精确复现当前发布 checkpoint")
    parser.add_argument('--max_images', type=int, default=None,
                        help="最多处理的图像数；少于目录总数时从完整序列中均匀抽取")
    parser.add_argument('--no_preview', action='store_true',
                        help="禁用预览图")
    parser.add_argument('--use_z_postprocess', action='store_true',
                        help="使用Z轴后处理")
    parser.add_argument('--no_z_postprocess', action='store_true',
                        help="禁用Z轴后处理")
    parser.add_argument('--no_reference', action='store_true',
                        help="【消融实验】不使用参考图像，使用输入图像自身作为参考")

    parser.add_argument('--upsample_xy', type=float, default=1.0,
                        help="XY方向上采样倍率")
    parser.add_argument('--upsample_order', type=int, default=1, choices=[0, 1, 3],
                        help="上采样插值阶数")

    parser.add_argument('--output_format', type=str, default='png', choices=['png', 'tif', 'both'],
                        help="输出格式")

    # 【新增】Reference结构破坏参数
    parser.add_argument('--struct_block_size', type=int, default=16,
                        help="非重叠参考图 block 边长；16 精确复现论文配置 checkpoint")
    parser.add_argument(
        '--reference_preprocess', default='block_mean',
        choices=['legacy', 'block_mean', 'raw_mean', 'zero', 'self'],
        help=(
            '显式参考图协议：block_mean=均值融合后分块均值；raw_mean=仅均值融合；'
            'zero=全零第二通道；self=输入自身；legacy=保持旧参数行为'
        ),
    )
    parser.add_argument('--reference_selection', choices=['uniform', 'random'], default='uniform',
                        help='参考目录中的切片选择策略；人工筛选目录通常使用 uniform')
    parser.add_argument('--reference_seed', type=int, default=42,
                        help='reference_selection=random 时的随机种子')
    reference_destruction = parser.add_mutually_exclusive_group()
    reference_destruction.add_argument('--destroy_ref', action='store_true',
                                       help="显式启用参考图结构破坏（全局像素置换）")
    reference_destruction.add_argument('--no_destroy_ref', action='store_true',
                                       help="禁用参考图结构破坏（默认；保留融合图的可见结构）")
    parser.add_argument('--reference_mode', choices=['fused', 'single'], default='fused',
                        help="fused: 融合多张参考图；single: 只使用一张参考图")
    parser.add_argument('--single_reference_index', type=int, default=0,
                        help="single 模式使用参考目录中按自然顺序排列的第几个图像（从 0 开始）")
    parser.add_argument('--legacy_per_image_normalization', action='store_true',
                        help="兼容旧推理：逐图百分位归一化；会破坏模型预测的绝对亮度，默认禁用")

    # 【新增】Z轴上下文参数
    parser.add_argument('--z_sequence_length', type=int, default=8,
                        help="Z轴窗口大小 (默认: 8)")
    parser.add_argument('--no_z_context', action='store_true',
                        help="禁用Z轴上下文（默认启用）")
    parser.add_argument('--use_z_context', action='store_true',
                        help='Compatibility option; the released network ignores explicit Z context')
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU编号")
    parser.add_argument('--cpu', action='store_true',
                        help="使用CPU")

    parser.add_argument('--disable_progress_bar', action='store_true',
                        help='Disable tqdm output in Slurm logs')

    parser.add_argument('--no_photometric_calibration', action='store_true',
                        help='Export network-only images; default directory inference saves whole-volume reference-calibrated final output')

    args = parser.parse_args()

    if args.input_dir is None and not args.no_photometric_calibration:
        parser.error('Single-image input cannot fit a whole volume; use --no_photometric_calibration or directory mode')

    if args.input_dir is not None and args.output_dir is not None:
        pass
    elif args.input is not None and args.output is not None:
        pass
    else:
        print("错误: 请使用 --input_dir --output_dir 或 --input --output")
        sys.exit(1)

    return args


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # 无缓冲输出

    args = parse_args()

    print("=" * 60)
    print("vEMDestripe inference")
    print("=" * 60)

    device = 'cuda' if not args.cpu else 'cpu'

    # 【新增】Reference结构破坏参数
    destroy_ref_struct = args.destroy_ref
    
    # 【新增】Z轴上下文参数
    # vEMDestripe does not consume Z context. Keep it off to avoid loading
    # a whole volume and printing one warning for every image.
    use_z_context = bool(args.use_z_context and not args.no_z_context)

    # Z轴后处理配置
    use_z_postprocess = args.use_z_postprocess and not args.no_z_postprocess

    inferencer = VEMDestripeInferencer(
        model_path=args.model,
        device=device,
        gpu=args.gpu,
        image_size=args.image_size,
        inference_size=args.inference_size,
        overlap_ratio=args.overlap_ratio,
        use_sliding_window=not args.no_sliding_window,
        use_z_postprocess=use_z_postprocess,
        upsample_xy=args.upsample_xy,
        upsample_order=args.upsample_order,
        destroy_reference_struct=destroy_ref_struct,
        struct_block_size=args.struct_block_size,
        struct_num_keep_blocks=0,
        z_sequence_length=args.z_sequence_length,
        use_z_context=use_z_context,
        reference_mode=args.reference_mode,
        single_reference_index=args.single_reference_index,
        legacy_per_image_normalization=args.legacy_per_image_normalization,
        train_compatible_reference=True,
        disable_progress_bar=args.disable_progress_bar,
        reference_preprocess=args.reference_preprocess,
        reference_selection=args.reference_selection,
        reference_seed=args.reference_seed,
    )

    use_sliding_window = not args.no_sliding_window

    if args.input_dir is not None:
        print(f"\n目录模式:")
        print(f"  输入目录: {args.input_dir}")
        print(f"  输出目录: {args.output_dir}")
        run_directory_pipeline(inferencer, args)
    else:
        print(f"\n单文件模式:")
        print(f"  输入文件: {args.input}")
        print(f"  输出文件: {args.output}")

        image = imread_grayscale(args.input)
        if image is None:
            print(f"错误: 无法读取图像 {args.input}")
            sys.exit(1)

        if args.reference is not None:
            reference = imread_grayscale(args.reference)
            reference = inferencer._destroy_reference(reference)
        else:
            print("错误: 单文件模式需要指定参考图像")
            sys.exit(1)

        result = inferencer.process_single(image, reference)
        imwrite_image(args.output, result)
        print(f"\n结果已保存: {args.output}")

    print("\n推理完成!")


if __name__ == "__main__":
    main()
