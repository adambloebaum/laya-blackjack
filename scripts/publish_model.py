"""Upload a verified model package to the authenticated user's private HF repository."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token

from blackjack.experiment_data import atomic_json, digest
from blackjack.releases import verify_model


def get_secret(name):
    value = os.environ.get(name)
    if value:
        return value
    env_file = Path.home() / ".Codex" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.match(rf"^\s*{re.escape(name)}\s*=\s*(.+)$", line)
            if match:
                return match.group(1).strip().strip("\"'")
    if os.name == "nt":
        try:
            return subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 f'[Environment]::GetEnvironmentVariable("{name}", "User")'],
                text=True, stderr=subprocess.DEVNULL,
            ).strip() or None
        except subprocess.CalledProcessError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", required=True, help="Personal owner/model repository")
    parser.add_argument("--version", required=True)
    parser.add_argument("--pin", type=Path, help="Write the verified commit and manifest digest here")
    args = parser.parse_args()
    manifest = verify_model(args.folder)
    token = next(
        (v for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
         if (v := get_secret(name))), None,
    ) or get_token()
    if not token:
        raise SystemExit("Missing Hugging Face credentials. Run hf auth login first.")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    response = session.get("https://huggingface.co/api/whoami-v2", timeout=30)
    response.raise_for_status()
    owner, name = args.repo.split("/")
    if response.json()["name"] != owner:
        raise SystemExit("Authenticated identity must match the personal repository owner.")
    url = f"https://huggingface.co/api/models/{args.repo}"
    response = session.get(url, timeout=30)
    if response.status_code == 404:
        response = session.post(
            "https://huggingface.co/api/repos/create",
            json={"type": "model", "name": name, "private": True}, timeout=30,
        )
        response.raise_for_status()
        response = session.get(url, timeout=30)
    response.raise_for_status()
    if response.json().get("private") is not True:
        raise SystemExit("This preparation tool only uploads to private model repositories.")
    commit = HfApi(token=token).upload_folder(
        repo_id=args.repo,
        folder_path=args.folder,
        allow_patterns=[*manifest["files"], "manifest.json"],
        commit_message=f"Release {args.version}: selected blackjack checkpoint and evaluation evidence",
    )
    # Read the committed bytes back before configuring the project's download pin.
    response = session.get(
        f"https://huggingface.co/{args.repo}/resolve/{commit.oid}/manifest.json", timeout=60,
    )
    response.raise_for_status()
    if response.content != (args.folder / "manifest.json").read_bytes():
        raise RuntimeError("Uploaded manifest does not match the prepared package.")
    release = {
        "repo_id": args.repo, "revision": commit.oid,
        "manifest_sha256": digest(args.folder / "manifest.json"), "version": args.version,
    }
    if args.pin:
        atomic_json(args.pin, release)
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
