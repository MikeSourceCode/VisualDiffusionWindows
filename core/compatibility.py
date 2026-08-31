"""Model compatibility checks for VisualDiffusion.

This module is the single source of truth for whether a model can be loaded
by the app. It is used by:
  - ``app.py``       -- local model validation before pipeline load

A model is considered compatible when it:
  * declares a Stable Diffusion / SDXL pipeline class in ``model_index.json``
  * uses CLIP-based text encoders / tokenizers
  * has the expected weight-file layout for our loading paths
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class CompatibilityResult:
    def __init__(self, compatible: bool, reasons: List[str]) -> None:
        self.compatible = compatible
        self.reasons = reasons

    def __bool__(self) -> bool:
        return self.compatible


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _walk_weight_files(root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith((".safetensors", ".bin")):
                out.append(os.path.join(dirpath, f))
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_model_index(model_index_path: Optional[str]) -> Tuple[bool, List[str]]:
    """Validate the model_index.json metadata."""
    reasons: List[str] = []
    if not model_index_path or not os.path.exists(model_index_path):
        reasons.append("model_index.json is missing")
        return False, reasons

    idx = _load_json(model_index_path)
    if idx is None:
        reasons.append("model_index.json is not valid JSON")
        return False, reasons

    class_name = str(idx.get("_class_name", ""))
    if not class_name:
        reasons.append("model_index.json is missing _class_name")
        return False, reasons

    supported_pipelines = {
        "StableDiffusionPipeline",
        "StableDiffusionXLPipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionXLImg2ImgPipeline",
    }
    if class_name not in supported_pipelines:
        reasons.append(
            f"Unsupported pipeline class: {class_name}. "
            "Only Stable Diffusion / SDXL pipelines are supported."
        )
        return False, reasons

    # Text encoder must be CLIP-based.
    text_encoder = idx.get("text_encoder", [])
    if isinstance(text_encoder, list) and len(text_encoder) >= 2:
        te_cls = str(text_encoder[1]).lower()
        if "clip" not in te_cls and "openclip" not in te_cls:
            reasons.append(
                f"Unsupported text encoder: {text_encoder[1]}. "
                "Only CLIP-based text encoders are supported."
            )
            return False, reasons

    tokenizer = idx.get("tokenizer", [])
    if isinstance(tokenizer, list) and len(tokenizer) >= 2:
        tok_cls = str(tokenizer[1]).lower()
        if "clip" not in tok_cls and "openclip" not in tok_cls:
            reasons.append(
                f"Unsupported tokenizer: {tokenizer[1]}. "
                "Only CLIP-based tokenizers are supported."
            )
            return False, reasons

    return True, reasons


def check_model_set_structure(model_path: str) -> Tuple[bool, List[str]]:
    """Validate that a model-set folder has the expected weight files."""
    reasons: List[str] = []
    if not os.path.isdir(model_path):
        reasons.append(f"Model set folder missing: {model_path}")
        return False, reasons

    index_path = os.path.join(model_path, "model_index.json")
    idx_ok, idx_reasons = check_model_index(index_path)
    reasons.extend(idx_reasons)
    if not idx_ok:
        return False, reasons

    # Look for weights in standard subfolders.
    candidates = ["unet", "transformer", "text_encoder", "tokenizer", "vae", "scheduler"]
    found_weights = False
    for name in candidates:
        d = os.path.join(model_path, name)
        if os.path.isdir(d) and _walk_weight_files(d):
            found_weights = True
            break

    if not found_weights:
        reasons.append(
            "No weight files (.safetensors/.bin) found in expected subfolders "
            "(unet/ or transformer/). Files may be missing from the model set."
        )
        return False, reasons

    return True, reasons


def check_backend_compatibility(model_path: str, backend: str) -> Tuple[bool, List[str]]:
    """Warn if the model is known to be CUDA-only on non-CUDA backends."""
    reasons: List[str] = []

    if backend == "mps":
        index_path = os.path.join(model_path, "model_index.json")
        idx = _load_json(index_path)
        if idx:
            class_name = str(idx.get("_class_name", ""))
            if "MageFlow" in class_name:
                reasons.append(
                    "MageFlow is currently CUDA-only and may not run on MPS."
                )
                return False, reasons

    return True, reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_local_model_set(model_path: str, backend: str = "cpu") -> CompatibilityResult:
    """Validate a local model set for use."""
    reasons: List[str] = []

    ok, idx_reasons = check_model_set_structure(model_path)
    reasons.extend(idx_reasons)
    if not ok:
        return CompatibilityResult(False, reasons)

    backend_ok, backend_reasons = check_backend_compatibility(model_path, backend)
    reasons.extend(backend_reasons)
    if not backend_ok:
        return CompatibilityResult(False, reasons)

    return CompatibilityResult(True, reasons)
