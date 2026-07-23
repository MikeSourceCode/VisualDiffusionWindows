#!/usr/bin/env python3
"""Download models from Hugging Face into the project's model folders.

Single-file checkpoint:
    python getmodel.py microsoft/Mage-Flow mage_flow.safetensors
    -> downloads that file into models/checkpoints/

Full model set (multi-file):
    python getmodel.py stabilityai/stable-diffusion-xl-base-1.0
    -> downloads the repo snapshot into models/model_set/stable-diffusion-xl-base-1.0/
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = REPO_ROOT / "models" / "checkpoints"
DEFAULT_MODEL_SET = REPO_ROOT / "models" / "model_set"


def _check_hf_cli() -> bool:
    return shutil.which("hf") is not None


def _run_hf_download(repo_id: str, local_dir: Path, filename: Optional[str] = None) -> None:
    cmd = ["hf", "download", repo_id, "--local-dir", str(local_dir)]
    if filename:
        cmd += ["--filename", filename]
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download(repo_id: str, filename: Optional[str] = None) -> None:
    if not _check_hf_cli():
        print("Error: 'hf' CLI is not installed. Install it with:")
        print("  curl -LsSf https://hf.co/cli/install.sh | bash")
        sys.exit(1)

    if filename:
        dest = DEFAULT_CHECKPOINTS
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading single file '{filename}' from '{repo_id}' -> {dest}")
        _run_hf_download(repo_id, dest, filename=filename)
    else:
        safe_name = repo_id.replace("/", "_")
        dest = DEFAULT_MODEL_SET / safe_name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading full model snapshot from '{repo_id}' -> {dest}")
        _run_hf_download(repo_id, dest)

    print("Done.")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    repo_id = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else None
    download(repo_id, filename=filename)


if __name__ == "__main__":
    main()
