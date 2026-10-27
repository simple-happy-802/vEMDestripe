"""Single-Original demo target, with production and evaluator kept separate."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import tifffile

from colab import original_target_demo as demo
from vemdestripe.calibration import calibration_lut, fit_affine, reference_statistics


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


@pytest.fixture
def small_demo(tmp_path, monkeypatch):
    root = tmp_path / 'assets'
    target = (np.arange(24, dtype=np.uint8).reshape(4, 6) * 3) + 30
    write_png(root / 'input/slice_00793.png', target)
    raw = np.stack([
        np.arange(24, dtype=np.uint8).reshape(4, 6) + 20,
        np.arange(24, dtype=np.uint8).reshape(4, 6) + 80,
    ])
    for offset, z in enumerate((777, 778)):
        write_png(root / 'precomputed/vEMDestripe' / f'slice_{z:05d}.png', raw[offset])
    # A deliberately different full-volume distribution prevents accidental
    # re-fitting from the two display sections from passing this fixture.
    frozen = {
        'raw_directory': 'precomputed/vEMDestripe',
        'source_statistics': {'mean': 120., 'std': 40., 'section_count': 2651,
                              'pixel_count': 2651 * 24, 'shape_yx': [4, 6]},
        'source_histograms_sha256': 'a' * 64,
    }
    monkeypatch.setattr(demo, 'frozen_calibration_record', lambda _: frozen)
    for method in ('EMISAC', 'FECNet', 'SSCOR'):
        for offset, z in enumerate((777, 778)):
            write_png(root / 'precomputed' / method / f'slice_{z:05d}.png', raw[offset])
    return root, target, raw, frozen


def test_frozen_target_is_original793_and_source_is_complete_volume(small_demo, tmp_path):
    root, target, raw, frozen = small_demo
    protected = {p: p.read_bytes() for p in root.rglob('*.png')}
    output = tmp_path / 'final'
    record = demo.materialize_original_target_pipeline(root, output)

    target_stats = reference_statistics([target])
    expected_scale, expected_bias = fit_affine((120., 40.),
                                              (target_stats['mean'], target_stats['std']))
    assert record['target_source'] == 'single_original_slice'
    assert record['target_section_id'] == 793
    assert record['target_statistics']['section_count'] == 1
    assert record['target_statistics']['mean'] == pytest.approx(float(target.mean()))
    assert record['target_statistics']['std'] == pytest.approx(float(target.std(ddof=0)))
    assert record['target_file']['name'] == 'slice_00793.png'
    assert record['target_file']['sha256'] == demo.sha256_file(root / 'input/slice_00793.png')
    assert record['source_statistics'] == frozen['source_statistics']
    raw_paths = sorted((root / 'precomputed/vEMDestripe').glob('slice_*.png'))
    assert record['raw_exported_files'] == [
        {'name': path.name, 'sha256': demo.sha256_file(path)} for path in raw_paths]
    assert len(record['raw_exported_files']) == 2
    assert record['source_statistics']['section_count'] == 2651
    assert demo.verify_exported_raw_files(record, root / 'precomputed/vEMDestripe')
    assert record['scale'] == expected_scale
    assert record['bias'] == expected_bias
    assert record['production_reference_pool_default_changed'] is False
    assert record['evaluator_photometric_transform'] is False
    assert record['complete'] is True

    lut = calibration_lut(expected_scale, expected_bias)
    np.testing.assert_array_equal(record['lut'], lut)
    exported = []
    for offset, z in enumerate((777, 778)):
        array = np.asarray(Image.open(output / f'slice_{z:05d}.png'))
        np.testing.assert_array_equal(array, lut[raw[offset]])
        exported.append(array)
    np.testing.assert_array_equal(tifffile.imread(output / 'vEMDestripe_final_demo.tif'), exported)
    saved_record = json.loads((output / 'CALIBRATION.json').read_text(encoding='utf-8'))
    assert saved_record['target_section_id'] == 793
    assert saved_record['output_stack']['sha256'] == demo.sha256_file(output / 'vEMDestripe_final_demo.tif')
    assert all(p.read_bytes() == before for p, before in protected.items())


def test_live_source_fits_all_live_sections_not_frozen_statistics(small_demo, tmp_path, monkeypatch):
    root, target, _, _ = small_demo
    live = tmp_path / 'live'
    arrays = [np.arange(24, dtype=np.uint8).reshape(4, 6) + i for i in (5, 40, 90)]
    for z, image in enumerate(arrays, 777):
        write_png(live / f'slice_{z:05d}_deflickered.png', image)
    def frozen_must_not_be_used(_):
        raise AssertionError('A fresh live volume must fit its own complete source')
    monkeypatch.setattr(demo, 'frozen_calibration_record', frozen_must_not_be_used)
    record = demo.materialize_original_target_pipeline(root, tmp_path / 'live_final', source_dir=live)
    complete = np.stack(arrays).astype(np.float64)
    assert record['source_statistics']['section_count'] == 3
    assert record['source_statistics']['mean'] == pytest.approx(float(complete.mean()))
    assert record['source_statistics']['std'] == pytest.approx(float(complete.std(ddof=0)))
    assert record['target_statistics']['section_count'] == 1
    assert record['target_statistics']['mean'] == pytest.approx(float(target.mean()))
    assert record['source_histograms_sha256'] is None
    assert record['exported_sections'] == 3


def test_rerun_refuses_overwrite(small_demo, tmp_path):
    root, _, _, _ = small_demo
    output = tmp_path / 'existing'
    demo.materialize_original_target_pipeline(root, output)
    first = (output / 'slice_00777.png').read_bytes()
    with pytest.raises(FileExistsError):
        demo.materialize_original_target_pipeline(root, output)
    assert (output / 'slice_00777.png').read_bytes() == first


@pytest.mark.parametrize('change', ['pixels', 'extra', 'missing'])
def test_stale_final_reuse_rejects_changed_raw_export(small_demo, tmp_path, change):
    root, _, _, _ = small_demo
    raw_dir = root / 'precomputed/vEMDestripe'
    record = demo.materialize_original_target_pipeline(root, tmp_path / 'final')
    assert demo.verify_exported_raw_files(record, raw_dir)
    if change == 'pixels':
        write_png(raw_dir / 'slice_00777.png', np.zeros((4, 6), dtype=np.uint8))
    elif change == 'extra':
        write_png(raw_dir / 'slice_00779.png', np.zeros((4, 6), dtype=np.uint8))
    else:
        (raw_dir / 'slice_00777.png').unlink()
    with pytest.raises(ValueError, match='raw export window changed'):
        demo.verify_exported_raw_files(record, raw_dir)


def test_geometry_mismatch_is_not_implicitly_resized(small_demo, tmp_path):
    root, _, _, _ = small_demo
    write_png(root / 'input/slice_00793.png', np.ones((6, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match='geometry'):
        demo.materialize_original_target_pipeline(root, tmp_path / 'bad_geometry')


def test_evaluation_sources_still_have_no_photometric_fitting():
    root = Path(__file__).parents[1]
    forbidden = {'fit_affine', 'calibration_lut', 'apply_calibration',
                 'match_to_target', 'matched_volumes', 'GlobalAffineStack'}
    for name in ('demo.py', 'evaluate.py', 'benchmark.py', 'niqe_cross_sections.py'):
        tree = ast.parse((root / 'evaluation' / name).read_text(encoding='utf-8'))
        calls = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not (calls & forbidden)
