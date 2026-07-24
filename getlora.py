#!/usr/bin/env python3
"""Download LoRAs from Hugging Face into models/lora/.

Usage:
    python getlora.py <repo_id> [filename]
    python getlora.py username/model-name my_lora.safetensors

If no filename is given and the repo contains exactly one .safetensors file,
it will be downloaded automatically.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LORA_DIR = REPO_ROOT / "models" / "lora"


def download(repo_id: str, filename: str | None = None) -> None:
    dest = DEFAULT_LORA_DIR
    dest.mkdir(parents=True, exist_ok=True)

    if filename:
        print(f"Downloading '{filename}' from '{repo_id}' -> {dest}")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[filename],
            local_dir=dest,
            local_dir_use_symlinks=False,
        )
    else:
        files = list(list_repo_files(repo_id))
        safetensors = [f for f in files if f.endswith(".safetensors")]
        if len(safetensors) == 1:
            auto_filename = safetensors[0]
            print(f"Detected single LoRA file: '{auto_filename}'")
            print(f"Downloading from '{repo_id}' -> {dest}")
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[auto_filename],
                local_dir=dest,
                local_dir_use_symlinks=False,
            )
        elif len(safetensors) > 1:
            print(f"Found {len(safetensors)} .safetensors files in '{repo_id}':")
            for f in safetensors:
                print(f"  - {f}")
            print("\nSpecify one with:")
            print(f"  python getlora.py {repo_id} <filename>.safetensors")
            sys.exit(1)
        else:
            print(f"No .safetensors files found in '{repo_id}'.")
            print("Falling back to full snapshot download...")
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
