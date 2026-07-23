"""Asset discovery: scan model folders for checkpoints, model sets, VAEs, LoRAs, controlnets.

A checkpoint is a single weight file in ``models/checkpoints/``.
A model set is a directory under ``models/model_set/`` that contains a full
diffusers-compatible snapshot (folder-based model).
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List

from .config import config_model_dirs


WEIGHT_EXTS = (".safetensors", ".pt", ".bin")


def detect_architecture(path: str) -> str:
    """Heuristically classify a checkpoint file as 'SDXL' or 'SD 1.5'."""
    if not path.lower().endswith(".safetensors"):
        return "SD 1.5"
    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        keys = header.keys()
        if any(k.startswith(("text_encoder_2", "conditioner")) for k in keys):
            return "SDXL"
        for k in keys:
            if k.endswith("attn2.to_k.weight"):
                return "SDXL" if header[k]["shape"][-1] == 2048 else "SD 1.5"
    except Exception:
        pass
    return "SD 1.5"


def _scan(folder: str) -> List[str]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(WEIGHT_EXTS))


def scan_checkpoints(folder: str) -> Dict[str, dict]:
    """Return {filename: {'path', 'arch'}} for every checkpoint file."""
    out: Dict[str, dict] = {}
    for f in _scan(folder):
        path = os.path.join(folder, f)
        out[f] = {"path": path, "arch": detect_architecture(path)}
    return out


def scan_subfolder(folder: str) -> List[str]:
    return _scan(folder)


def _scan_model_set(folder: str) -> Dict[str, dict]:
    """Recursively scan ``models/model_set`` for full-model directories.

    Only immediate children of ``folder`` are considered candidates. A candidate
    is treated as a model set if any weight file (``.safetensors`` / ``.bin``)
    or ``model_index.json`` exists anywhere inside its subtree.
    """
    out: Dict[str, dict] = {}
    if not os.path.isdir(folder):
        return out
    candidates = sorted(
        d for d in os.listdir(folder)
        if not d.startswith(".") and os.path.isdir(os.path.join(folder, d))
    )
    for name in candidates:
        root = os.path.join(folder, name)
        has_weights = False
        model_index_path = None
        for r, _, files in os.walk(root):
            for f in files:
                if f.endswith(WEIGHT_EXTS):
                    has_weights = True
                elif f == "model_index.json" and model_index_path is None:
                    model_index_path = os.path.join(r, f)
                if has_weights and model_index_path:
                    break
            if has_weights and model_index_path:
                break
        if has_weights or model_index_path:
            out[name] = {"path": root, "arch": _detect_model_set_arch(root, model_index_path)}
    return out


def _detect_model_set_arch(root: str, model_index_path: Optional[str]) -> str:
    """Heuristically classify a model-set directory as 'SDXL' or 'SD 1.5'."""
    if os.path.isdir(os.path.join(root, "text_encoder_2")):
        return "SDXL"
    if model_index_path and os.path.exists(model_index_path):
        try:
            with open(model_index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            repo = idx.get("repo_id", "") or ""
            if "xl" in repo.lower() or "sdxl" in repo.lower():
                return "SDXL"
        except Exception:
            pass
    return "SD 1.5"


def discover_all(models_dir: str | None = None) -> Dict[str, object]:
    """Return a dict of every asset category discovered on disk."""
    d = config_model_dirs(models_dir)
    checkpoints = scan_checkpoints(d["checkpoints"])
    model_sets = _scan_model_set(d["model_set"])
    return {
        "dirs": d,
        "checkpoints": checkpoints,
        "model_sets": model_sets,
        "vaes": scan_subfolder(d["vae"]),
        "loras": scan_subfolder(d["lora"]),
        "controlnets": scan_subfolder(d["controlnet"]),
        "ip_adapters": scan_subfolder(d["ip_adapter"]),
    }
