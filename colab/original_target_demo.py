"""Demo-only vEMDestripe calibration to the user's fixed Original section.

This changes the demonstration target, not the production reference-pool
pipeline. Baselines and evaluators never call a calibration transform.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tifffile

try:
    from .reference_pipeline_demo import calibration_record as frozen_calibration_record
    from .official_demo import sha256_file
    from .demo_metrics import load_png_volume
except ImportError:
    from reference_pipeline_demo import calibration_record as frozen_calibration_record
    from official_demo import sha256_file
    from demo_metrics import load_png_volume
from vemdestripe.calibration import (reference_statistics, volume_statistics,
    read_uint8_image, calibration_record as make_record)

TARGET_SECTION_ID = 793
TARGET_RELATIVE_PATH = 'input/slice_00793.png'


def _raw_export_inventory(source_dir):
    return [{'name': path.name, 'sha256': sha256_file(path)}
            for path in sorted(Path(source_dir).glob('slice_*.png'))]


def verify_exported_raw_files(record, source_dir):
    """Reject stale final files if their supplied raw export window changed.

    This inventory covers only the exported raw PNGs, not the complete 2651
    sections represented by the frozen source-statistics histograms.
    """
    expected = record.get('raw_exported_files')
    if not expected or expected != _raw_export_inventory(source_dir):
        raise ValueError('Network raw export window changed; use a fresh result directory or runtime')
    return True


def materialize_original_target_pipeline(root, output_dir, source_dir=None):
    """Save vEM output using ONE map to Original793; no per-section fitting.

    Frozen mode retains complete 2651-section source statistics. Optional
    source_dir fits all newly supplied live output sections, explicitly a
    separate volume. The target is the original input image, not a clean label.
    """
    root, output = Path(root), Path(output_dir)
    target_path = root / TARGET_RELATIVE_PATH
    target_image = read_uint8_image(target_path)
    target = reference_statistics([target_image])
    if source_dir is None:
        frozen = frozen_calibration_record(root)
        raw_dir = root / frozen['raw_directory']
        source = dict(frozen['source_statistics'])
        context = 'complete 2651-section saved network output; apply one map then crop display window'
        source_histogram_sha = frozen['source_histograms_sha256']
    else:
        raw_dir = Path(source_dir)
        source = None
        context = 'all supplied live demo output sections; not the published 2651-section fit'
        source_histogram_sha = None
    raw, ids = load_png_volume(raw_dir)
    if raw.shape[1:] != target_image.shape:
        raise ValueError('Original target and output section geometry differ')
    if source is None:
        source = volume_statistics(raw)
    record = make_record(source, target, target_source='single_original_slice')
    record.update(target_section_id=TARGET_SECTION_ID,
        target_file={'name':target_path.name, 'relative_path':TARGET_RELATIVE_PATH,
                     'sha256':sha256_file(target_path)},
        target_role='user-selected Original demonstration input; not a clean ground-truth claim',
        only_method_calibrated='vEMDestripe', demo_target_override=True,
        production_reference_pool_default_changed=False, source_context=context,
        source_histograms_sha256=source_histogram_sha,
        exported_sections=len(ids), exported_local_z=[ids[0],ids[-1]+1],
        raw_exported_files=_raw_export_inventory(raw_dir),
        raw_inventory_scope='exported raw PNG window only; separate from complete source statistics',
        evaluator_photometric_transform=False, evaluation_normalization=False)
    output.mkdir(parents=True, exist_ok=True)
    marker = output/'CALIBRATION.json'
    stack_path = output/'vEMDestripe_final_demo.tif'
    if marker.exists() or stack_path.exists() or list(output.glob('slice_*.png')):
        raise FileExistsError('Choose a new demo output directory; existing final files are not overwritten')
    final = np.asarray(record['lut'],dtype=np.uint8)[raw]
    output_files=[]
    for z,image in zip(ids,final):
        path=output/f'slice_{z:05d}.png'
        Image.fromarray(image).save(path)
        if not np.array_equal(read_uint8_image(path),image):
            raise ValueError('Saved PNG pixels differ from the declared fixed mapping')
        output_files.append({'name':path.name,'sha256':sha256_file(path)})
    tifffile.imwrite(stack_path,final,photometric='minisblack',metadata={'axes':'ZYX'})
    if not np.array_equal(tifffile.imread(stack_path),final):
        raise ValueError('Final TIFF differs from exported PNG pixels')
    record.update(complete=True,output_files=output_files,output_stack={
        'name':stack_path.name,'sha256':sha256_file(stack_path)})
    marker.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return record
