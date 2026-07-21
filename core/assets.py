"""Asset discovery: scan model folders for checkpoints, VAEs, LoRAs, controlnets.

Uses the same `.safetensors` header inspection the original scripts used to
detect SDXL vs SD 1.5.
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


def discover_all(models_dir: str | None = None) -> Dict[str, object]:
    """Return a dict of every asset category discovered on disk."""
    d = config_model_dirs(models_dir)
    checkpoints = scan_checkpoints(d["checkpoints"])
    return {
        "dirs": d,
        "checkpoints": checkpoints,
        "vaes": scan_subfolder(d["vae"]),
        "loras": scan_subfolder(d["lora"]),
        "controlnets": scan_subfolder(d["controlnet"]),
        "ip_adapters": scan_subfolder(d["ip_adapter"]),
    }
