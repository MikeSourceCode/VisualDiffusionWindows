#!/usr/bin/env python3
"""Standalone Mage-Flow runner.

This script attempts to load and run microsoft/Mage-Flow as a standalone
pipeline, outside of the VisualDiffusion Streamlit app.

Requirements:
  - A diffusers version that registers MageFlowPipeline
  - The model set downloaded to models/model_set/microsoft_Mage-Flow/

Usage:
  python scripts/run_mage_flow.py
  python scripts/run_mage_flow.py --prompt "your prompt here"
  python scripts/run_mage_flow.py --steps 30 --seed 123
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from diffusers import DiffusionPipeline


def detect_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device(backend: str) -> torch.device:
    if backend == "cuda":
        return torch.device("cuda")
    if backend == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_path() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "model_set" / "microsoft_Mage-Flow",
        PROJECT_ROOT / "models" / "model_set" / "microsoft-mage-flow",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fallback: first model_set subdir
    model_set_root = PROJECT_ROOT / "models" / "model_set"
    if model_set_root.exists():
        for child in sorted(model_set_root.iterdir()):
            if child.is_dir() and "mage" in child.name.lower():
                return child
    return candidates[0]


def print_model_index(path: Path):
    index = path / "model_index.json"
    if not index.exists():
        print("[warn] No model_index.json found")
        return
    with open(index, "r", encoding="utf-8") as f:
        idx = json.load(f)
    print(f"  _class_name: {idx.get('_class_name')}")
    print(f"  transformer: {idx.get('transformer')}")
    print(f"  vae: {idx.get('vae')}")
    print(f"  text_encoder: {idx.get('text_encoder')}")
    print(f"  scheduler: {idx.get('scheduler')}")


def load_pipeline(model_path: Path, device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading Mage-Flow from: {model_path}")
    print("Model index:")
    print_model_index(model_path)
    print(f"\nAttempting DiffusionPipeline.from_pretrained() ...")

    try:
        pipe = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
        )
    except AttributeError as exc:
        raise RuntimeError(
            f"MageFlowPipeline is not available in the installed diffusers version.\n"
            f"Original error: {exc}\n"
            "To fix this, either:\n"
            "  1. Upgrade diffusers: pip install -U diffusers\n"
            "  2. Install a package that provides MageFlowPipeline\n"
            "  3. Use a Stable Diffusion / SDXL model set instead"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Mage-Flow model set: {type(exc).__name__}: {exc}"
        ) from exc

    return pipe


def run_generation(pipe, prompt: str, negative: str, steps: int, seed: int, device: str):
    print(f"\nPrompt: {prompt}")
    print(f"Negative: {negative}")
    print(f"Steps: {steps}, Seed: {seed}, Device: {device}")

    generator = torch.Generator("cpu").manual_seed(seed) if seed >= 0 else None
    t0 = time.time()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
    except Exception as exc:
        print(f"\nGeneration failed: {type(exc).__name__}: {exc}")
        return None

    elapsed = time.time() - t0
    print(f"Generation finished in {elapsed:.1f}s")
    return result


def save_result(result, output_dir: Path):
    if result is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    out_path = output_dir / f"mage_flow_{timestamp}.png"
    result.save(out_path)
    print(f"Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mage-Flow standalone")
    parser.add_argument("--prompt", default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path()

    if not model_path.exists():
        print(f"ERROR: Model set not found: {model_path}")
        print("Download it first with:")
        print(f"  python getmodel.py microsoft/Mage-Flow")
        sys.exit(1)

    backend = detect_backend()
    device = str(get_device(backend))

    print("=" * 70)
    print("Mage-Flow standalone runner")
    print("=" * 70)
    print(f"Backend: {backend}")
    print(f"Device: {device}")
    print(f"Model: {model_path.name}")

    try:
        pipe = load_pipeline(model_path, device)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print(f"\nLoaded pipeline: {type(pipe).__name__}")
    result = run_generation(pipe, args.prompt, args.negative, args.steps, args.seed, device)
    save_result(result, args.output_dir)

    # Clean up
    del pipe
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
