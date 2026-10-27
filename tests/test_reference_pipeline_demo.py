import ast
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from evaluation.demo import load_saved_methods,score_volumes
from evaluation.evaluate import prepare_saved_outputs,METHOD_CACHE_SCHEMA
from colab.demo_metrics import METHOD_ORDER
from colab import reference_pipeline_demo as bundle


def test_evaluator_cannot_fit_or_apply_photometric_mapping():
    for name in ['demo.py','evaluate.py']:
        tree=ast.parse((Path(__file__).parents[1]/'evaluation'/name).read_text(encoding='utf-8'))
        names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert not names & {'match_to_target','matched_volumes','fit_affine','calibration_lut',
                            'GlobalAffineStack','apply_calibration','global_affine_match'}
    assert METHOD_CACHE_SCHEMA==3


def test_window_reads_delivered_values_unchanged_and_gt_is_only_target():
    class Saved:
        count=4
        def read(self,i):return np.full((4,4),i+20,np.uint8)
        def read_many(self,ids):return np.stack([self.read(i) for i in ids])
    source=Saved();stacks={m:source for m in METHOD_ORDER}
    selected,target=prepare_saved_outputs(stacks,METHOD_ORDER,1,3,source)
    for m in METHOD_ORDER:
        assert selected[m].source is source
        np.testing.assert_array_equal(selected[m].read(0),source.read(1))
    assert target.source is source


def test_saved_demo_loading_and_scoring_do_not_change_images(tmp_path,monkeypatch):
    from evaluation import demo
    from colab import demo_metrics
    dirs={}
    for i,m in enumerate(METHOD_ORDER):
        folder=tmp_path/m;folder.mkdir();dirs[m]=folder
        for z in [777,778]:Image.fromarray(np.full((32,32),40+i*10+z%2,np.uint8)).save(folder/f'slice_{z:05d}.png')
    volumes,ids=load_saved_methods(dirs,expected_ids=[777,778])
    before={m:a.copy() for m,a in volumes.items()}
    def forbidden(*a,**k):raise AssertionError('Evaluator must not normalize')
    monkeypatch.setattr(demo_metrics,'matched_volumes',forbidden)
    monkeypatch.setattr(demo_metrics,'global_affine_match',forbidden)
    monkeypatch.setattr(demo.metrics,'niqe_score',lambda *a,**k:1.)
    monkeypatch.setattr(demo.metrics,'_niqe_orientation',lambda *a,**k:1.)
    model=tmp_path/'prior.npz';np.savez(model,mu=np.zeros(36),cov=np.eye(36))
    result=score_volumes(volumes,model,xy_samples=1,cross_sections=1,border=0)
    assert not result['evaluator_photometric_transform'].any()
    for m in METHOD_ORDER:np.testing.assert_array_equal(volumes[m],before[m])


def test_v3_extractor_rejects_traversal_before_creating_output(tmp_path):
    import zipfile
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr(bundle.BUNDLE_NAME+'/../escape','bad')
    with pytest.raises(bundle.BundleValidationError):
        bundle.safe_extract_bundle(path,tmp_path/'extracted',bundle.sha256_file(path))
    assert not (tmp_path/'escape').exists()
