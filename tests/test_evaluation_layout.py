from pathlib import Path

import cv2
import numpy as np

from evaluation.benchmark import ground_truth, image_sequence


def write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((16, 16), value, dtype=np.uint8))


def test_nested_input_and_groundtruth_layout(tmp_path):
    write_png(tmp_path / "test_data" / "sample" / "test" / "striped" / "slice_000.png", 20)
    write_png(tmp_path / "test_data" / "sample" / "groundtruth" / "slice_000.png", 30)
    original = image_sequence(tmp_path / "test_data", "sample")
    target = ground_truth(tmp_path, "sample")
    assert original.count == 1
    assert target is not None and target.count == 1
    assert int(original.read(0).mean()) == 20
    assert int(target.read(0).mean()) == 30

