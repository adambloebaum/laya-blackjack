import json
import sys
from types import ModuleType

import pytest

from blackjack.experiment_data import atomic_json
from blackjack.releases import fetch_model, install_model, package_model, public_metadata, verify_model


def package_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"test inventory bytes; not a real model")
    atomic_json(source / "rl_agent_config.json", {"max_len": 1024, "head_max_len": 256})
    atomic_json(
        source / "training_report.json", {"source": "/home/private/checkpoint", "test": {"agreement": 0.9}}
    )
    atomic_json(source / "encoder/config.json", {})
    atomic_json(source / "tokenizer/tokenizer.json", {})
    for name in ("LICENSE", "NOTICE", "MODEL_CARD.md"):
        (tmp_path / name).write_text("Test documentation\n")
    target = tmp_path / "package"
    package_model(source, target, tmp_path / "MODEL_CARD.md", tmp_path)
    return target


def test_package_installs_atomically_and_rejects_tampering(tmp_path):
    package = package_fixture(tmp_path)
    installed = install_model(package, tmp_path / "installed")
    assert install_model(package, installed) == installed
    assert verify_model(installed)["format"] == "laya-blackjack-checkpoint-v1"
    assert (
        json.loads((installed / "training_report.json").read_text())["source"] == "local-artifact:checkpoint"
    )
    (package / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        install_model(package, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_model_inventory_rejects_traversal_and_wrong_state_contract(tmp_path):
    package = package_fixture(tmp_path)
    path = package / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = {"bytes": 1, "sha256": "0" * 64}
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="Unsafe path"):
        verify_model(package)
    manifest["state_contract"] = "different"
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="incompatible"):
        verify_model(package)


def test_public_metadata_preserves_hashes_and_removes_local_paths():
    assert public_metadata({"hash": "abc", "source": ["/home/someone/model", "convaiinnovations/laya"]}) == {
        "hash": "abc",
        "source": ["local-artifact:model", "convaiinnovations/laya"],
    }


def test_release_packaging_rejects_sealed_test_candidates(tmp_path):
    package = package_fixture(tmp_path)
    atomic_json(package / "training_report.json", {"test": None, "test_deferred": True})
    with pytest.raises(ValueError, match="Final-test evaluation"):
        package_model(package, tmp_path / "release", tmp_path / "MODEL_CARD.md", tmp_path)


def test_pinned_download_only_requests_manifest_files(tmp_path, monkeypatch):
    package = package_fixture(tmp_path)
    module = ModuleType("huggingface_hub")
    revision = "a" * 40
    calls = []

    def download(repo, filename, **kwargs):
        assert repo == "owner/model" and kwargs["revision"] == revision
        assert filename == "manifest.json"
        return str(package / filename)

    def snapshot(repo, **kwargs):
        assert repo == "owner/model" and kwargs["revision"] == revision
        calls.append(kwargs)
        return str(package)

    module.hf_hub_download = download
    module.snapshot_download = snapshot
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    output = fetch_model(tmp_path / "downloaded", "owner/model", revision)
    assert verify_model(output)
    assert set(calls[0]["allow_patterns"]) == set(verify_model(package)["files"]) | {"manifest.json"}
    with pytest.raises(ValueError, match="40-character"):
        fetch_model(tmp_path / "bad", "owner/model", "main")
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["files"]["../escape"] = {"bytes": 0, "sha256": "0" * 64}
    atomic_json(package / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Unsafe path"):
        fetch_model(tmp_path / "bad", "owner/model", revision)
    assert len(calls) == 1
