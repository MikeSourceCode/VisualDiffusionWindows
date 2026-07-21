"""Model catalog: the single source of truth for downloadable assets.

Both setup.py (interactive download) and the app (runtime "is it present?"
checks and slow-load notices) read from this list so paths never drift.

Each entry describes one asset:
- key:     stable identifier
- kind:    checkpoint | vae | controlnet | annotator
- repo:    Hugging Face repo id
- files:   list of filenames to fetch (empty => whole-repo snapshot)
- dest:    directory under the project (relative)
- size:    approximate download size, human string (for the prompt)
- desc:    short description
- required_for: which app feature(s) need it
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class CatalogEntry:
    key: str
    kind: str
    repo: str
    dest: str
    desc: str
    size: str
    files: List[str] = field(default_factory=list)
    required_for: str = ""

    def dest_abs(self, root: str) -> str:
        return os.path.join(root, self.dest)

    def is_present(self, root: str) -> bool:
        """Heuristic presence check: destination exists and holds weight files."""
        if self.kind == "annotator":
            return _annotator_cached(self.repo)
        d = self.dest_abs(root)
        if not os.path.isdir(d):
            return False
        if self.files:
            return all(os.path.exists(os.path.join(d, os.path.basename(f))) for f in self.files)
        # Snapshot-style (annotator/controlnet dir): any weight file present.
        for _root, _dirs, fnames in os.walk(d):
            if any(f.endswith((".safetensors", ".bin", ".pth", ".ckpt", ".pt")) for f in fnames):
                return True
        return False


CATALOG: List[CatalogEntry] = [
    CatalogEntry(
        key="sdxl_base",
        kind="checkpoint",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        files=["sd_xl_base_1.0.safetensors"],
        dest="models/checkpoints",
        size="~6.9 GB",
        desc="SDXL Base 1.0 (official, SFW default checkpoint)",
        required_for="all tabs",
    ),
    CatalogEntry(
        key="sdxl_vae",
        kind="vae",
        repo="stabilityai/sdxl-vae",
        files=["sdxl_vae.safetensors"],
        dest="models/vae",
        size="~335 MB",
        desc="SDXL VAE (optional; SDXL base has a built-in VAE)",
        required_for="optional",
    ),
    CatalogEntry(
        key="controlnet_canny_sdxl",
        kind="controlnet",
        repo="diffusers/controlnet-canny-sdxl-1.0",
        files=["config.json", "diffusion_pytorch_model.safetensors"],
        dest="models/controlnet/canny-sdxl-1.0",
        size="~2.5 GB",
        desc="ControlNet Canny SDXL (Image Control tab / PyraCanny)",
        required_for="Image Control tab",
    ),
    CatalogEntry(
        key="controlnet_openpose_sdxl",
        kind="controlnet",
        repo="xinsir/controlnet-openpose-sdxl-1.0",
        files=["config.json", "diffusion_pytorch_model.safetensors"],
        dest="models/controlnet/openpose-sdxl-1.0",
        size="~2.5 GB",
        desc="ControlNet OpenPose SDXL (Pose Control tab)",
        required_for="Pose Control tab",
    ),
    CatalogEntry(
        key="openpose_annotator",
        kind="annotator",
        repo="lllyasviel/Annotators",
        files=[],  # controlnet_aux caches these in the HF cache, not a project dir
        dest="",   # cache-based; presence is checked via _annotator_cached()
        size="~200 MB",
        desc="OpenPose annotator weights (detect pose FROM an image)",
        required_for="Pose Control tab (image mode)",
    ),
]


def _annotator_cached(repo: str = "lllyasviel/Annotators") -> bool:
    """Check whether the controlnet_aux annotator weights exist in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # body_pose_model.pth is the core OpenPose body file.
        hit = try_to_load_from_cache(repo, "body_pose_model.pth")
        return isinstance(hit, str) and os.path.exists(hit)
    except Exception:
        return False


def by_key(key: str) -> CatalogEntry | None:
    for e in CATALOG:
        if e.key == key:
            return e
    return None


def missing_entries(root: str) -> List[CatalogEntry]:
    return [e for e in CATALOG if not e.is_present(root)]
