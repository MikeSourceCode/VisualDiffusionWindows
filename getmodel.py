#!/usr/bin/env python3
"""Download models from Hugging Face into the project's model folders.

Single-file checkpoint:
    python getmodel.py microsoft/Mage-Flow mage_flow.safetensors
    -> downloads that file into models/checkpoints/

    Or, if the repo has exactly one .safetensors file, omit the filename:
    python getmodel.py OnomaAIResearch/Illustrious-XL-v2.0
    -> auto-detects the checkpoint and downloads to models/checkpoints/

Full model set (multi-file):
    python getmodel.py stabilityai/stable-diffusion-xl-base-1.0
    -> downloads the repo snapshot into models/model_set/stable-diffusion-xl-base-1.0/
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = REPO_ROOT / "models" / "checkpoints"
DEFAULT_MODEL_SET = REPO_ROOT / "models" / "model_set"


def _check_repo_compatibility(repo_id: str) -> bool:
    """Validate model_index.json via HF Hub API before downloading."""
    try:
        from core.compatibility import validate_repo
    except ImportError:
        print("[warn] core.compatibility not available; skipping pre-download check")
        return True

    print(f"Checking compatibility for '{repo_id}' ...")
    result = validate_repo(repo_id)
    if result:
        print("  -> Compatible")
        return True

    print("  -> INCOMPATIBLE:")
    for reason in result.reasons:
        print(f"     - {reason}")
    return False


def _is_single_file_checkpoint_repo(repo_id: str) -> tuple[bool, list[str]]:
    """Check if a repo is a single-file checkpoint repo.

    Returns (is_single_file, list_of_safetensors_files).
    """
    try:
        files = list(list_repo_files(repo_id))
        safetensors = [f for f in files if f.endswith(".safetensors")]
        return len(safetensors) == 1, safetensors
    except Exception:
        return False, []


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
        # Check if this is a model set with model_index.json
        compat_ok = _check_repo_compatibility(repo_id)
        if compat_ok:
            safe_name = repo_id.replace("/", "_")
            dest = DEFAULT_MODEL_SET / safe_name
            dest.mkdir(parents=True, exist_ok=True)
            print(f"Downloading full model snapshot from '{repo_id}' -> {dest}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest,
                local_dir_use_symlinks=False,
            )
        else:
            # model_index.json missing or incompatible — maybe it's a single-file checkpoint?
            is_single, safetensors_files = _is_single_file_checkpoint_repo(repo_id)
            if is_single and safetensors_files:
                auto_filename = safetensors_files[0]
                print(f"\nDetected single-file checkpoint: '{auto_filename}'")
                print(f"Downloading to {DEFAULT_CHECKPOINTS} ...")
                dest = DEFAULT_CHECKPOINTS
                dest.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[auto_filename],
                    local_dir=dest,
                    local_dir_use_symlinks=False,
                )
            else:
                print("\nAborting download. This repo is not a compatible model set.")
                if safetensors_files:
                    print(f"Found {len(safetensors_files)} .safetensors files:")
                    for f in safetensors_files:
                        print(f"  - {f}")
                    print("\nSpecify one with:")
                    print(f"  python getmodel.py {repo_id} <filename>.safetensors")
                else:
                    print("Use a Stable Diffusion / SDXL model set instead.")
                sys.exit(1)
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
