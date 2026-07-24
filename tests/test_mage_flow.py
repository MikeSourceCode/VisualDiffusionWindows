"""Standalone console test for the Mage-Flow model set.

This is intentionally NOT integrated into the Streamlit app. It exists so we
can experiment with loading microsoft/Mage-Flow without affecting the main
pipeline code paths.

Current status:
  - The model set download IS complete (transformer weights present).
  - MageFlow uses flow matching (rectified flows), not the noise-prediction
    architecture used by Stable Diffusion / SDXL.
  - diffusers 0.39.0 does not include MageFlowPipeline.

Usage:
    python -m tests.test_mage_flow
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SET_DIR = PROJECT_ROOT / "models" / "model_set" / "microsoft_Mage-Flow"
PROMPT = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
NEGATIVE = ""
STEPS = 20
SEED = 42


def detect_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def inspect_model_set(model_path: Path):
    print("\n=== Model set inspection ===")
    if not model_path.exists():
        print(f"  Directory does not exist: {model_path}")
        return

    files = list(model_path.rglob("*"))
    weight_files = [f for f in files if f.is_file() and f.suffix in (".safetensors", ".bin")]
    print(f"  Total files: {len([f for f in files if f.is_file()])}")
    print(f"  Weight files: {len(weight_files)}")
    for w in weight_files[:10]:
        print(f"    {w.relative_to(model_path)}")
    if len(weight_files) > 10:
        print(f"    ... and {len(weight_files) - 10} more")

    index_path = model_path / "model_index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        print(f"\n  model_index.json:")
        print(f"    _class_name: {idx.get('_class_name')}")
        print(f"    transformer: {idx.get('transformer')}")
        print(f"    vae: {idx.get('vae')}")
        print(f"    text_encoder: {idx.get('text_encoder')}")
        print(f"    tokenizer: {idx.get('tokenizer')}")
        print(f"    scheduler: {idx.get('scheduler')}")


def try_load_as_sdxl(model_path: Path, device: str):
    print("\n=== Attempt 1: load as SDXL ===")
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            use_safetensors=True,
        )
        print(f"  SUCCESS: {type(pipe).__name__}")
        return pipe
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return None


def try_load_as_sd15(model_path: Path, device: str):
    print("\n=== Attempt 2: load as SD 1.5 ===")
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            use_safetensors=True,
        )
        print(f"  SUCCESS: {type(pipe).__name__}")
        return pipe
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return None


def try_diffusion_pipeline(model_path: Path, device: str):
    print("\n=== Attempt 3: generic DiffusionPipeline ===")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        pipe = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
        )
        print(f"  SUCCESS: {type(pipe).__name__}")
        return pipe
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return None


def run():
    print("=" * 70)
    print("Mage-Flow standalone test")
    print("=" * 70)

    inspect_model_set(MODEL_SET_DIR)

    if not MODEL_SET_DIR.exists():
        print(f"\nModel set not found: {MODEL_SET_DIR}")
        print("Download it first with:")
        print(f"  python getmodel.py microsoft/Mage-Flow")
        sys.exit(1)

    device = detect_backend()
    print(f"\nBackend: {device}")
    print(f"diffusers version: {DiffusionPipeline.__module__}")

    pipe = try_load_as_sdxl(MODEL_SET_DIR, device)
    if pipe is None:
        pipe = try_load_as_sd15(MODEL_SET_DIR, device)
    if pipe is None:
        pipe = try_diffusion_pipeline(MODEL_SET_DIR, device)

    if pipe is None:
        print("\n" + "=" * 70)
        print("RESULT: Cannot load Mage-Flow with current diffusers installation.")
        print("=" * 70)
        print("\nRoot cause:")
        print("  The model declares _class_name='MageFlowPipeline' in model_index.json,")
        print("  but diffusers 0.39.0 does not include this pipeline class.")
        print("  The model also uses a custom transformer (mage_flow.MageFlow)")
        print("  and text encoder (Qwen3VLForConditionalGeneration) that are not")
        print("  compatible with Stable Diffusion / SDXL pipeline classes.")
        print("\nTo make this work, you would need one of:")
        print("  1. Upgrade diffusers to a version that registers MageFlowPipeline")
        print("  2. Install a custom Microsoft package that provides MageFlowPipeline")
        print("  3. Implement a MageFlowAdapter in this project")
        sys.exit(1)

    print(f"\nLoaded: {type(pipe).__name__}")
    print("Attempting generation...")
    generator = torch.Generator("cpu").manual_seed(SEED) if SEED >= 0 else None
    t0 = time.time()
    try:
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            num_inference_steps=STEPS,
            generator=generator,
        ).images[0]
        elapsed = time.time() - t0
        out_path = PROJECT_ROOT / "outputs" / f"mage_flow_test_{int(time.time())}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        print(f"Saved: {out_path} ({elapsed:.1f}s)")
    except Exception as exc:
        print(f"Generation failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run()
