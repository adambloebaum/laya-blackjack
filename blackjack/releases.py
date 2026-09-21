"""Portable model packages, pinned downloads, and atomic local installation."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath

from .experiment_data import atomic_json, digest
from .model import BASE_MODEL, BASE_REVISION, model_state, questions

FORMAT = "laya-blackjack-checkpoint-v1"
REQUIRED = {
    "model.safetensors",
    "rl_agent_config.json",
    "training_report.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "README.md",
    "LICENSE",
    "NOTICE",
}


def state_contract():
    return hashlib.sha256(
        (inspect.getsource(model_state) + inspect.getsource(questions)).encode()
    ).hexdigest()


def public_metadata(value):
    """Remove machine-specific paths from reports while retaining artifact identities."""
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, str) and value.startswith(("/home/", "/tmp/", "/mnt/")):
        return "local-artifact:" + Path(value).name
    return value


def verify_model(root: Path, expected_manifest_sha256=None):
    manifest_path = root / "manifest.json"
    if expected_manifest_sha256 and digest(manifest_path) != expected_manifest_sha256:
        raise ValueError("Release manifest does not match the pinned digest.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("state_contract") != state_contract():
        raise ValueError("Model package uses an incompatible blackjack state/question contract.")
    files = manifest.get("files", {})
    if not REQUIRED <= set(files):
        raise ValueError("Model package is missing required inference or documentation files.")
    for name, info in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
            raise ValueError("Unsafe path in model inventory.")
        target = root / name
        if not target.is_file() or target.stat().st_size != info["bytes"] or digest(target) != info["sha256"]:
            raise ValueError(f"Model file failed integrity verification: {name}")
    config = json.loads((root / "rl_agent_config.json").read_text())
    if config.get("max_len") != 1024 or config.get("head_max_len") != 256:
        raise ValueError("Checkpoint context budgets do not match serving.")
    return manifest


def package_model(source: Path, output: Path, card: Path, project: Path, evidence: Path | None = None):
    if output.exists():
        raise ValueError("Use a new package directory; released files are immutable.")
    if not (source / "training_report.json").exists():
        raise ValueError("Only a completed, evaluated checkpoint can be packaged.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".staging-" + uuid.uuid4().hex[:8])
    staging.mkdir()
    for name in ("model.safetensors", "rl_agent_config.json"):
        shutil.copyfile(source / name, staging / name)
    for name in ("encoder", "tokenizer"):
        shutil.copytree(source / name, staging / name)
    atomic_json(
        staging / "training_report.json",
        public_metadata(json.loads((source / "training_report.json").read_text())),
    )
    shutil.copyfile(card, staging / "README.md")
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(project / name, staging / name)
    if evidence:
        shutil.copytree(evidence, staging / "evaluation")
    manifest = {
        "format": FORMAT,
        "state_contract": state_contract(),
        "upstream": {"repo_id": BASE_MODEL, "revision": BASE_REVISION},
        "files": {
            str(path.relative_to(staging)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        },
    }
    atomic_json(staging / "manifest.json", manifest)
    verify_model(staging)
    staging.rename(output)
    return manifest


def install_model(source: Path, output: Path, expected_manifest_sha256=None):
    manifest = verify_model(source, expected_manifest_sha256)
    if output.exists():
        verify_model(output, digest(source / "manifest.json"))
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".staging-" + uuid.uuid4().hex[:8])
    staging.mkdir()
    for name in [*manifest["files"], "manifest.json"]:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    verify_model(staging, digest(source / "manifest.json"))
    staging.rename(output)
    return output


def fetch_model(output: Path, repo: str | None = None, revision: str | None = None):
    from huggingface_hub import hf_hub_download, snapshot_download

    expected = None
    if repo is None and revision is None:
        release_path = Path(__file__).with_name("model_release.json")
        if not release_path.exists():
            raise ValueError("No published release is configured yet; supply --repo and --revision.")
        release = json.loads(release_path.read_text())
        repo, revision, expected = release["repo_id"], release["revision"], release["manifest_sha256"]
    if not repo or not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Specify a Hugging Face repository and its full 40-character commit revision.")
    manifest_path = Path(hf_hub_download(repo, "manifest.json", revision=revision))
    manifest = json.loads(manifest_path.read_text())
    if expected and digest(manifest_path) != expected:
        raise ValueError("Downloaded release manifest does not match the pinned digest.")
    files = list(manifest.get("files", {}))
    if manifest.get("format") != FORMAT or not REQUIRED <= set(files):
        raise ValueError("Repository does not contain a complete blackjack model package.")
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name
        for name in files
    ):
        raise ValueError("Unsafe path in remote model inventory.")
    snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=[*files, "manifest.json"]))
    return install_model(snapshot, output, expected or digest(manifest_path))
