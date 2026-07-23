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
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = REPO_ROOT / "models" / "checkpoints"
DEFAULT_MODEL_SET = REPO_ROOT / "models" / "model_set"


def download(repo_id: str, filename: str | None = None) -> None:
    if filename:
        dest = DEFAULT_CHECKPOINTS
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading '{filename}' from '{repo_id}' -> {dest}")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[filename],
            local_dir=dest,
            local_dir_use_symlinks=False,
        )
    else:
        safe_name = repo_id.replace("/", "_")
        dest = DEFAULT_MODEL_SET / safe_name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading full model snapshot from '{repo_id}' -> {dest}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            local_dir_use_symlinks=False,
        )
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
