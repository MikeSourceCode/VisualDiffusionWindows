"""Central configuration model for the Visual Diffusion app.

A single ``AppConfig`` instance is the source of truth for all generation
settings. The universal sidebar writes to it and every feature tab reads from
it, which prevents per-tab setting drift.

Config file format (JSON, ``config/app_config.json``):

    {
      "common": {
        "checkpoint": "sd_xl_base_1.0.safetensors",
        "vae": "",
        "loras": [],
        "steps": 20,
        "cfg_scale": 7.0,
        ...
      },
      "tabs": {
        "text_to_image": {
          "default_prompt": "...",
          "default_negative_prompt": "..."
        },
        "image_to_image": {
          "strength": 0.65,
          "default_prompt": "...",
          "default_negative_prompt": "..."
        }
      }
    }

``common`` populates the shared ``AppConfig``. ``tabs`` holds per-tab overrides
(prompts, tab-specific defaults) that are NOT exposed in the sidebar but are
read by the tab code at runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# LoRA spec = (filename, weight). Weight 0.0 disables the adapter.
LoraSpec = Tuple[str, float]


def config_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "config")


def config_file_path() -> str:
    return os.path.join(config_dir(), "app_config.json")


def load_config_file(path: Optional[str] = None) -> Dict[str, Any]:
    """Load operator config from JSON. Returns empty dict if missing/invalid."""
    path = path or config_file_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config_file(data: Dict[str, Any], path: Optional[str] = None) -> None:
    path = path or config_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


@dataclass
class AppConfig:
    """Global generation configuration shared across every feature tab.

    The object is plain-data so it can be (de)serialized for preset saving and
    reproduced exactly by the generation engine.
    """

    # --- Checkpoint / VAE (file selection) ---
    checkpoint: str = ""           # filename inside models/checkpoints
    vae: str = ""                  # "" => built-in VAE
    architecture: str = "SDXL"      # "SDXL" | "SD 1.5" (derived from checkpoint)

    # --- LoRAs (up to 2 stackable) ---
    loras: List[LoraSpec] = field(default_factory=list)

    # --- Global hyperparameters ---
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = 0                  # 0 => random
    sharpness: float = 0.0         # 0.0 => off
    strength: float = 0.65         # img2img denoise strength

    # --- Output size (txt2img). Default is 9:16 (TikTok/Reels/Shorts), matching
    # the standalone scripts. For img2img the init image drives the size.
    # NOTE: diffusers requires dimensions divisible by 8; 1368 (not 1366) is the
    # nearest 9:16-multiple of the requested 768x1366.
    width: int = 768
    height: int = 1368

    # --- Preview / UX ---
    show_previews: bool = True
    preview_frequency: int = 1

    # --- Safety (operator-configurable via config file) ---
    # Early safety gate: run the CLIP NSFW check ONCE at a fraction of denoise
    # progress instead of every step (slow) or only at the end (wasted compute).
    # If it flags, generation aborts early. The final output guard (safety.censor)
    # still always runs.
    early_safety_check: bool = True
    # Fixed step at which to run the early safety check (e.g. 2 = after step 2).
    # If 0, falls back to early_safety_step_frac.
    early_safety_step: int = 0
    # Fraction of total steps at which to run the early safety check when
    # early_safety_step is 0. Lower = earlier abort on unsafe content.
    # Default 0.05 = after ~5% of denoising (e.g. step 1 of 20).
    early_safety_step_frac: float = 0.05

    # --- ControlNet / conditioning extras (used by specific tabs) ---
    controlnet_strength: float = 1.0
    ip_adapter_scale: float = 0.6

    # --- Prompt defaults (operator-configurable; replaces hardcoded per-tab
    # defaults in prompt_with_tags when set in config file). ---
    default_prompt: str = ""
    default_negative_prompt: str = ""

    # --- Internal: per-tab overrides from config file (not in to_dict). ---
    _tabs_config: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def lora_specs(self) -> List[LoraSpec]:
        """Return only the enabled LoRA specs (named and weight > 0)."""
        return [(n, w) for (n, w) in self.loras if n and w > 0.0]

    def preview_every(self, total_steps: int) -> int:
        """Clamp preview frequency to a sane value for the step count."""
        if not self.preview_frequency or self.preview_frequency < 1:
            return 1
        return min(self.preview_frequency, max(1, total_steps))

    def effective_seed(self) -> int:
        """Materialize the seed; 0 (or negative) becomes a fresh random seed."""
        import random
        return self.seed if self.seed and self.seed > 0 else random.randint(1, 2**31)

    def tab_config(self, tab_name: str) -> Dict[str, Any]:
        """Return per-tab overrides from the config file (empty dict if none)."""
        return self._tabs_config.get(tab_name, {})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        cfg = cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        for key, value in data.items():
            if key == "tabs":
                cfg._tabs_config = value if isinstance(value, dict) else {}
                continue
            if key not in known:
                continue
            if key == "loras" and isinstance(value, list):
                value = [tuple(spec) if isinstance(spec, (list, tuple)) else spec for spec in value]
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_config_file(cls, path: Optional[str] = None) -> "AppConfig":
        """Load config from JSON file, falling back to defaults if missing."""
        data = load_config_file(path)
        if not data:
            return cls()
        common = data.get("common", data)
        merged = dict(common)
        merged["tabs"] = data.get("tabs", {})
        return cls.from_dict(merged)


def default_models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def config_model_dirs(models_dir: Optional[str] = None):
    models_dir = models_dir or default_models_dir()
    return {
        "models": models_dir,
        "checkpoints": os.path.join(models_dir, "checkpoints"),
        "vae": os.path.join(models_dir, "vae"),
        "lora": os.path.join(models_dir, "lora"),
        "controlnet": os.path.join(models_dir, "controlnet"),
        "ip_adapter": os.path.join(models_dir, "ip-adapter"),
    }
