"""Network-free provenance checks for the official EMISAC launcher."""

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "colab" / "run_official_emisac.py"
SPEC = importlib.util.spec_from_file_location("official_emisac_launcher", MODULE_PATH)
emisac = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(emisac)


def test_rejects_tracked_edits_before_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(emisac, "_git_commit", lambda _: emisac.OFFICIAL_COMMIT)
    monkeypatch.setattr(emisac, "_capture", lambda *args, **kwargs: " M EM_normalization.py")
    with pytest.raises(emisac.EmisacCompatibilityError, match="tracked modifications"):
        emisac.patch_official_source(tmp_path)


def test_verify_requires_both_source_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(emisac, "_git_commit", lambda _: emisac.OFFICIAL_COMMIT)
    manifest = {
        "compatibility_version": emisac.COMPATIBILITY_VERSION,
        "official_commit": emisac.OFFICIAL_COMMIT,
        "sha256_before": {},
        "sha256_after": {},
    }
    (tmp_path / emisac.PATCH_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(emisac.EmisacCompatibilityError, match="both official script hashes"):
        emisac.verify_prepared_source(tmp_path)


def test_verify_detects_modified_prepared_source(tmp_path, monkeypatch):
    monkeypatch.setattr(emisac, "_git_commit", lambda _: emisac.OFFICIAL_COMMIT)
    names = ("EM_normalization.py", "normalize_partition.py")
    for name in names:
        (tmp_path / name).write_text("original content\n", encoding="utf-8")
    hashes = {name: emisac._sha256(tmp_path / name) for name in names}
    manifest = {
        "compatibility_version": emisac.COMPATIBILITY_VERSION,
        "official_commit": emisac.OFFICIAL_COMMIT,
        "sha256_before": hashes,
        "sha256_after": hashes,
    }
    (tmp_path / emisac.PATCH_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / names[0]).write_text("modified content\n", encoding="utf-8")
    with pytest.raises(emisac.EmisacCompatibilityError, match="hash mismatch"):
        emisac.verify_prepared_source(tmp_path)
