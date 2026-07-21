"""Visual Diffusion core package.

Shared generation engine, configuration model, asset discovery and local
state persistence used by the Streamlit app and the standalone scripts.
"""

from .config import AppConfig, LoraSpec, config_model_dirs
from .assets import discover_all, scan_checkpoints, scan_subfolder, detect_architecture
from .tags import load_tag_groups, tags_to_prompt_fragment
from .preprocess import pyracanny, canny, preprocess, PREPROCESSORS
from .caption import caption_image, is_available as caption_available
from .pose import (
    detect_pose_from_image,
    parse_openpose_json,
    render_openpose_skeleton,
    skeleton_from_json,
)
from .engine import (
    Backend,
    VRAMState,
    detect_backend,
    detect_vram_mb,
    get_vram_state,
    load_base_pipeline,
    load_controlnet_pipeline,
    load_loras_into_unet,
    generate,
    default_vae_path,
    load_init_image_from_path,
    place_unet,
    place_vae,
    place_clip,
)

__all__ = [
    "AppConfig",
    "LoraSpec",
    "Backend",
    "VRAMState",
    "config_model_dirs",
    "discover_all",
    "scan_checkpoints",
    "scan_subfolder",
    "detect_architecture",
    "detect_backend",
    "detect_vram_mb",
    "get_vram_state",
    "load_base_pipeline",
    "load_controlnet_pipeline",
    "load_loras_into_unet",
    "generate",
    "default_vae_path",
    "load_init_image_from_path",
    "place_unet",
    "place_vae",
    "place_clip",
    "load_tag_groups",
    "tags_to_prompt_fragment",
    "pyracanny",
    "canny",
    "preprocess",
    "PREPROCESSORS",
    "caption_image",
    "caption_available",
    "detect_pose_from_image",
    "parse_openpose_json",
    "render_openpose_skeleton",
    "skeleton_from_json",
]
