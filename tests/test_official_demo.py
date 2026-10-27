"""Integrity and raw-output protocol tests for the public official-result demo."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab"))
import official_demo as demo


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "original" / demo.BUNDLE_NAME
    root.mkdir(parents=True)
    sources = {}
    for name in demo.OFFICIAL_SOURCE_SHA256:
        path = root / demo.PROVENANCE / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test fixture: " + name, encoding="utf-8")
        sources[name] = demo.sha256_file(path)
    monkeypatch.setattr(demo, "OFFICIAL_SOURCE_SHA256", sources)
    record = {"original_source_unmodified": True, "official_commit": demo.OFFICIAL_COMMIT,
              "input_local_z": [777, 809], "shape_zyx": list(demo.EXPECTED_SHAPE)}
    (root / demo.PROVENANCE / "metadata.json").write_text(json.dumps(record), encoding="utf-8")
    for name in ("actual_run.json", "original.log", "normalized_data_final.h5", "input_stack.tif"):
        (root / demo.PROVENANCE / name).write_bytes(b"fixture")
    np.savez(root / "bumbarger13_niqe_model.npz", mu=np.zeros(36), cov=np.eye(36))
    manifest = {"schema_version": 2, "bundle_name": demo.BUNDLE_NAME,
                "input_local_z": [777, 809], "shape_zyx": list(demo.EXPECTED_SHAPE),
                "files": {p.relative_to(root).as_posix(): demo.sha256_file(p)
                          for p in root.rglob("*") if p.is_file()}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def make_zip(root, target):
    with zipfile.ZipFile(target, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, f"{root.name}/{path.relative_to(root).as_posix()}")
    return target


def test_manifest_and_tamper_detection(bundle):
    assert demo.verify_bundle(bundle)["bundle_name"] == demo.BUNDLE_NAME
    (bundle / demo.PROVENANCE / "original.log").write_bytes(b"tampered")
    with pytest.raises(demo.BundleValidationError, match="checksum mismatch"):
        demo.verify_bundle(bundle)


def test_unlisted_file_is_rejected(bundle):
    (bundle / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(demo.BundleValidationError, match="Unlisted files"):
        demo.verify_bundle(bundle)


def test_official_commit_and_unmodified_flag(bundle):
    path = bundle / demo.PROVENANCE / "metadata.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["original_source_unmodified"] = "true"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(demo.BundleValidationError, match="unmodified"):
        demo.official_metadata(bundle)


def test_safe_extract_and_idempotent_verified_reuse(bundle, tmp_path):
    archive = make_zip(bundle, tmp_path / "demo.zip")
    expected = demo.sha256_file(archive)
    extracted = demo.safe_extract_bundle(archive, tmp_path / "download", expected)
    assert extracted.name == demo.BUNDLE_NAME
    assert demo.safe_extract_bundle(archive, tmp_path / "download", expected) == extracted
    with pytest.raises(demo.BundleValidationError, match="ZIP checksum"):
        demo.safe_extract_bundle(archive, tmp_path / "other", "0" * 64)


@pytest.mark.parametrize("entry", ["../outside", "/absolute", "C:/drive", "x\\escape", f"{demo.BUNDLE_NAME}/../escape"])
def test_unsafe_zip_paths_rejected(tmp_path, entry):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(entry, "bad")
    with pytest.raises(demo.BundleValidationError):
        demo.safe_extract_bundle(archive_path, tmp_path / "out", demo.sha256_file(archive_path))
    assert not (tmp_path / "outside").exists()


def test_zip_symlink_rejected(tmp_path):
    archive_path = tmp_path / "link.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        info = zipfile.ZipInfo(f"{demo.BUNDLE_NAME}/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside")
    with pytest.raises(demo.BundleValidationError, match="symbolic link"):
        demo.safe_extract_bundle(archive_path, tmp_path / "out", demo.sha256_file(archive_path))


def test_loader_checks_section_ids_and_shape(bundle, monkeypatch):
    monkeypatch.setattr(demo, "verify_bundle", lambda root: {})
    monkeypatch.setattr(demo.metrics, "load_png_volume", lambda folder: (np.zeros((2, 3, 3), np.uint8), [777, 778]))
    with pytest.raises(demo.BundleValidationError, match="expected local Z"):
        demo.load_official_demo(bundle)


def test_raw_evaluation_never_matches_implicitly(tmp_path, monkeypatch):
    model = tmp_path / "model.npz"
    np.savez(model, mu=np.zeros(36), cov=np.eye(36))
    base = np.tile(np.arange(64, dtype=np.uint8)[None, None, :], (4, 64, 1))
    volumes = {name: base + i * 10 for i, name in enumerate(demo.metrics.METHOD_ORDER)}
    monkeypatch.setattr(demo.metrics, "niqe_score", lambda *args: float(np.mean(args[0])))
    monkeypatch.setattr(demo.metrics, "_niqe_orientation", lambda *args: 0.0)
    monkeypatch.setattr(demo.metrics, "matched_volumes", lambda *args: pytest.fail("Raw evaluation performed hidden matching"))
    table = demo.evaluate_outputs(volumes, model, xy_samples=2, cross_sections=1, border=0)
    assert table["output_mode"].eq("raw").all()
    assert table["global_match_scale"].eq(1.0).all()
    assert table["global_match_bias"].eq(0.0).all()
    assert table.loc["EMISAC", "NIQE_XY"] != table.loc["Original", "NIQE_XY"]
    assert not any("PSNR" in name or "SSIM" in name for name in table.columns)


def test_both_protocols_are_separately_labelled(tmp_path, monkeypatch):
    model = tmp_path / "model.npz"
    np.savez(model, mu=np.zeros(36), cov=np.eye(36))
    base = np.tile(np.arange(64, dtype=np.uint8)[None, None, :], (4, 64, 1))
    volumes = {name: base + i * 10 for i, name in enumerate(demo.metrics.METHOD_ORDER)}
    monkeypatch.setattr(demo.metrics, "niqe_score", lambda *args: 1.0)
    monkeypatch.setattr(demo.metrics, "_niqe_orientation", lambda *args: 1.0)
    table = demo.evaluate_both(volumes, model, xy_samples=1, cross_sections=1, border=0)
    assert len(table) == 10
    assert set(table["output_mode"]) == {"raw", "global_affine_matched"}
    assert table[table["output_mode"] == "raw"]["global_match_bias"].eq(0).all()
    assert table[(table["output_mode"] == "global_affine_matched")].loc["EMISAC", "global_match_bias"] != 0


@pytest.mark.parametrize("tamper", [False, True])
def test_h5_to_png_pixel_identity(tmp_path, monkeypatch, tamper):
    import h5py

    root = tmp_path / "bundle"
    h5_path = root / demo.PROVENANCE / "normalized_data_final.h5"
    h5_path.parent.mkdir(parents=True)
    raw = np.arange(36, dtype=np.float64).reshape(3, 3, 4) + 0.25
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("stack", data=raw)
    record = {"raw_h5_sha256": demo.sha256_file(h5_path)}
    monkeypatch.setattr(demo, "official_metadata", lambda root: record)
    monkeypatch.setattr(demo, "EXPECTED_SHAPE", (4, 3, 3))
    monkeypatch.setattr(demo, "EXPECTED_IDS", tuple(range(777, 781)))
    png = np.rint(raw).astype(np.uint8).transpose(2, 0, 1)
    if tamper:
        png = png.copy()
        png[0, 0, 0] += 1
    monkeypatch.setattr(demo.metrics, "load_png_volume", lambda folder: (png, list(range(777, 781))))
    if tamper:
        with pytest.raises(demo.BundleValidationError, match="not the declared"):
            demo.verify_emisac_pixels(root)
    else:
        assert demo.verify_emisac_pixels(root)["png_values_identical_to_raw_h5_quantization"] is True
