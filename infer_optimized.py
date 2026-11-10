"""Run fixed E1 inference and postprocessing on a directory of PNG sections."""
import argparse
from pathlib import Path

import cv2
import torch

from vemdestripe.optimized import DEFAULT_CHECKPOINT, run_volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=4, help="PNG I/O workers")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    record = run_volume(args.input_dir, args.output_dir, args.checkpoint, args.device, args.workers)
    print(f"Restored {record['shape_zyx'][0]} sections in {record['processing_seconds']:.3f} s")
    print(f"Results: {args.output_dir / 'restored'}")


if __name__ == "__main__":
    main()
