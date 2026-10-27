import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from vemdestripe.calibration import (calibration_lut,fit_affine,volume_statistics,
    reference_statistics,calibrate_directory,read_uint8_image)


def test_lut_exact_audited_float32_contract():
    for scale,bias in [(0.6342516813452413,-2.0677735015677996),(.5,.5),(1.2,-20),(.01,210)]:
        expected=np.rint(np.clip(np.arange(256,dtype=np.float32)*scale+bias,0,255)).astype(np.uint8)
        np.testing.assert_array_equal(calibration_lut(scale,bias),expected)


def test_complete_source_and_pooled_original_targets(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    first=np.arange(48,dtype=np.uint8).reshape(6,8)
    second=first+100
    for i,image in enumerate([first,second]):Image.fromarray(image).save(raw/f'slice_{i:05d}_deflickered.png')
    references=[first+20,first+40]
    paths=[]
    for i,image in enumerate(references):
        p=tmp_path/f'reference_{i}.png';Image.fromarray(image).save(p);paths.append(p)
    stats=volume_statistics([first,second]); target=reference_statistics(references)
    all_source=np.stack([first,second]).astype(float)
    assert stats['mean']==pytest.approx(all_source.mean())
    assert stats['std']==pytest.approx(all_source.std(ddof=0))
    assert target['mean']==pytest.approx(np.stack(references).mean())
    final=tmp_path/'final'
    record=calibrate_directory(raw,final,paths,expected_count=2)
    assert record['source_statistics']['section_count']==2
    assert record['target_statistics']['section_count']==2
    assert record['per_slice_fit'] is False
    lut=np.asarray(record['lut'],dtype=np.uint8)
    np.testing.assert_array_equal(read_uint8_image(raw/'slice_00000_deflickered.png'),first)
    np.testing.assert_array_equal(read_uint8_image(final/'slice_00001_deflickered.png'),lut[second])
    with pytest.raises(FileExistsError):calibrate_directory(raw,final,paths)
    with pytest.raises(ValueError,match='Incomplete'):calibrate_directory(raw,tmp_path/'incomplete',paths,expected_count=3)


def test_constants_and_non_uint8_are_not_silently_rescaled():
    assert fit_affine((4,0),(10,0))==(1.,6.)
    with pytest.raises(ValueError,match='constant'):fit_affine((4,0),(10,2))
    with pytest.raises(ValueError,match='uint8'):volume_statistics([np.ones((3,3),np.float32)])


def test_default_directory_pipeline_keeps_raw_and_saves_final(tmp_path):
    from infer import run_directory_pipeline
    references=tmp_path/'references';references.mkdir()
    ref=np.arange(48,dtype=np.uint8).reshape(6,8)+30
    Image.fromarray(ref).save(references/'original.png')
    class FakeNetwork:
        reference_preprocess='block_mean'
        def process_directory(self,**kwargs):
            out=Path(kwargs['output_dir']);out.mkdir(parents=True)
            for i in range(2):Image.fromarray(ref+i*20).save(out/f'slice_{i:05d}_deflickered.png')
            (out/'processing_info.json').write_text(json.dumps({'reference_indices':['original.png'],
                'num_input_images_total':2,'num_images':2}))
    args=SimpleNamespace(output_dir=str(tmp_path/'result'),no_photometric_calibration=False,
        no_reference=False,max_images=None,input_dir=str(tmp_path/'input'),reference=str(references),
        num_reference=16,no_preview=True,no_sliding_window=True,output_format='png',
        use_z_context=False,no_z_context=True,z_sequence_length=8)
    run_directory_pipeline(FakeNetwork(),args)
    root=Path(args.output_dir)
    assert (root/'network_raw/slice_00000_deflickered.png').is_file()
    assert (root/'slice_00000_deflickered.png').is_file()
    record=json.loads((root/'CALIBRATION.json').read_text())
    assert record['complete'] and record['source_statistics']['section_count']==2
    assert record['target_statistics']['mean']==pytest.approx(ref.mean())
    with pytest.raises(FileExistsError):run_directory_pipeline(FakeNetwork(),args)
