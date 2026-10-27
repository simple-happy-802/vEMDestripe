"""Verified transport and output materialization for the complete vEM pipeline.

Only vEMDestripe is calibrated. Evaluation is in evaluation/demo.py and reads
the final files after export; it never calls a fitting or calibration function.
The frozen 32-section display uses the recorded FULL 2651-section fit.
"""
from pathlib import Path
import json
import shutil
import stat
import zipfile
import numpy as np
from PIL import Image
import tifffile

try:
    from .official_demo import (sha256_file, _relative_path, _SHA256,
        MAX_UNCOMPRESSED_BYTES, BundleValidationError, official_metadata, verify_emisac_pixels)
    from .demo_metrics import METHOD_ORDER, load_png_volume
except ImportError:
    from official_demo import (sha256_file, _relative_path, _SHA256,
        MAX_UNCOMPRESSED_BYTES, BundleValidationError, official_metadata, verify_emisac_pixels)
    from demo_metrics import METHOD_ORDER, load_png_volume
from vemdestripe.calibration import (moments_from_histograms, reference_statistics,
    read_uint8_image, fit_affine, calibration_lut)

BUNDLE_NAME='vemdestripe_bumbarger_reference_pipeline_v3'
RECORD='calibration/vemdestripe_full_volume.json'
EXPECTED_IDS=list(range(777,809))
EXPECTED_SHAPE=(32,1024,1024)


def calibration_record(root, *, verify_pixels=True):
    """Verify the separately fitted full-volume mapping and its original targets."""
    root=Path(root)
    record=json.loads((root/RECORD).read_text(encoding='utf-8'))
    if (record['method']!='vEMDestripe' or record['only_method_calibrated']!='vEMDestripe'
        or record['source_scope']!='all_saved_output_voxels'
        or record['target_source']!='pooled_original_reference_snapshots'
        or record['source_statistics']['section_count']!=2651
        or record['target_statistics']['section_count']!=11):
        raise BundleValidationError('Unexpected calibration scope or target')
    path=root/record['source_histograms_path']
    if sha256_file(path)!=record['source_histograms_sha256']:
        raise BundleValidationError('Complete source histogram checksum mismatch')
    with np.load(path) as data:
        hist=data['histograms']
    if hist.shape!=(2651,256): raise BundleValidationError('Source statistics omit complete output sections')
    source=moments_from_histograms(hist)
    references=record['reference_files']
    for item in references:
        ref=root.joinpath(*_relative_path(item['path']).parts)
        if sha256_file(ref)!=item['sha256']: raise BundleValidationError('Original reference snapshot changed')
    target=reference_statistics(read_uint8_image(root/item['path']) for item in references)
    for actual,expected in zip(source,(record['source_statistics']['mean'],record['source_statistics']['std'])):
        if abs(actual-expected)>1e-12: raise BundleValidationError('Full source statistics disagree')
    if not np.allclose([target['mean'],target['std']],
        [record['target_statistics']['mean'],record['target_statistics']['std']],atol=1e-12,rtol=0):
        raise BundleValidationError('Target must be pooled original reference pixels')
    scale,bias=fit_affine(source,(target['mean'],target['std']))
    if not np.allclose([scale,bias],[record['scale'],record['bias']],atol=1e-12,rtol=0):
        raise BundleValidationError('Recorded mapping differs from the full-volume fit')
    lut=calibration_lut(scale,bias)
    if not np.array_equal(lut,np.asarray(record['lut'],dtype=np.uint8)):
        raise BundleValidationError('Recorded quantization differs from the audited LUT')
    if verify_pixels:
        raw,ids=load_png_volume(root/record['raw_directory'])
        final,final_ids=load_png_volume(root/record['final_directory'])
        if ids!=EXPECTED_IDS or final_ids!=ids or raw.shape!=EXPECTED_SHAPE or not np.array_equal(lut[raw],final):
            raise BundleValidationError('Final vEM images do not equal the declared full-volume calibration')
    return record


def verify_bundle(root):
    root=Path(root)
    if root.is_symlink() or not root.is_dir(): raise BundleValidationError('Invalid asset directory')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('bundle_name')!=BUNDLE_NAME or manifest.get('schema_version')!=3
        or manifest.get('calibrated_methods')!=['vEMDestripe']
        or manifest.get('input_local_z')!=[777,809]):
        raise BundleValidationError('Wrong complete-pipeline bundle')
    files=manifest.get('files')
    if not isinstance(files,dict) or not files: raise BundleValidationError('Missing file inventory')
    for name,expected in files.items():
        relative=_relative_path(name)
        path=root.joinpath(*relative.parts)
        if not isinstance(expected,str) or not _SHA256.fullmatch(expected): raise BundleValidationError('Invalid file checksum')
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise BundleValidationError('Unsafe or missing asset: '+name)
        if sha256_file(path)!=expected: raise BundleValidationError('Asset checksum mismatch: '+name)
    actual=set()
    for path in root.rglob('*'):
        if path.is_symlink(): raise BundleValidationError('Links are not accepted in the bundle')
        if path.is_file(): actual.add(path.relative_to(root).as_posix())
    if actual!=set(files)|{'manifest.json'}: raise BundleValidationError('Unexpected or missing bundle files')
    official_metadata(root)
    calibration_record(root)
    return manifest


def safe_extract_bundle(zip_path,destination,expected_sha256):
    archive_path,parent=Path(zip_path),Path(destination)
    if not _SHA256.fullmatch(expected_sha256) or sha256_file(archive_path)!=expected_sha256:
        raise BundleValidationError('Release archive checksum mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        entries,seen,total=[],set(),0
        for info in archive.infolist():
            name=info.filename.rstrip('/') if info.is_dir() else info.filename
            relative=_relative_path(name)
            kind=stat.S_IFMT(info.external_attr>>16)
            if relative.parts[0]!=BUNDLE_NAME or name in seen or kind not in (0,stat.S_IFREG,stat.S_IFDIR):
                raise BundleValidationError('Unsafe ZIP member')
            if len(relative.parts)==1 and not info.is_dir(): raise BundleValidationError('Expected one bundle folder')
            seen.add(name);total+=info.file_size
            if total>MAX_UNCOMPRESSED_BYTES: raise BundleValidationError('ZIP is too large')
            entries.append((info,relative))
        if not entries: raise BundleValidationError('Empty archive')
        parent.mkdir(parents=True,exist_ok=True)
        root=parent/BUNDLE_NAME
        if root.exists() or root.is_symlink(): verify_bundle(root);return root
        root.mkdir()
        try:
            for info,relative in entries:
                target=parent.joinpath(*relative.parts)
                if info.is_dir(): target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(info) as source,target.open('xb') as output:
                        shutil.copyfileobj(source,output,1024*1024)
            verify_bundle(root)
        except BaseException:
            shutil.rmtree(root)
            raise
    return root


def materialize_frozen_pipeline(root,output_dir):
    """Apply the recorded FULL-volume mapping, save final PNG/TIFF, then return.

    This is an output stage, not an evaluator. It does not fit the 32-section
    display. Full-source statistics are independently validated above.
    """
    root,output=Path(root),Path(output_dir)
    record=calibration_record(root)
    output.mkdir(parents=True,exist_ok=True)
    marker=output/'CALIBRATION.json'
    if marker.exists() or list(output.glob('slice_*.png')):
        raise FileExistsError('Select a new final output directory, or reuse a verified completed export')
    lut=np.asarray(record['lut'],dtype=np.uint8)
    raw,ids=load_png_volume(root/record['raw_directory'])
    expected,_=load_png_volume(root/record['final_directory'])
    final=lut[raw]
    if not np.array_equal(final,expected): raise BundleValidationError('Recorded final output mismatch')
    files=[]
    for image,z in zip(final,ids):
        path=output/f'slice_{z:05d}.png'
        Image.fromarray(image).save(path)
        files.append({'name':path.name,'sha256':sha256_file(path)})
    tifffile.imwrite(output/'vEMDestripe_final_demo.tif',final,photometric='minisblack',metadata={'axes':'ZYX'})
    record=dict(record,complete=True,exported_local_z=[777,809],exported_sections=32,
                fit_scope='complete 2651-section saved network output; not fitted to this display window',
                output_files=files,evaluation_normalization=False)
    marker.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return record


def method_directories(root,final_vem_dir):
    root=Path(root)
    return {'Original':root/'input','EMISAC':root/'precomputed/EMISAC',
            'FECNet':root/'precomputed/FECNet','SSCOR':root/'precomputed/SSCOR',
            'vEMDestripe':Path(final_vem_dir)}


def verify_baselines_unchanged(root,baseline_v2_root):
    """Author-side check that no delivered baseline received a new transform."""
    a,b=Path(root),Path(baseline_v2_root)
    checked=0
    for directory in ('input','precomputed/EMISAC','precomputed/FECNet','precomputed/SSCOR'):
        for path in (b/directory).glob('slice_*.png'):
            if sha256_file(a/directory/path.name)!=sha256_file(path):
                raise BundleValidationError('A delivered baseline changed')
            checked+=1
    return {'baseline_png_files_unchanged':checked,'additional_baseline_calibration':False}
