"""Image captioning (image -> text) using BLIP.

Used by the Image->Text->Image tab to describe an uploaded image so a brand-new
image can be generated from the description alone (a clean "IP wash": the output
shares no pixels with the source, only the described concept).

The model is loaded lazily and cached at module level so repeated calls in a
Streamlit session do not reload weights. Runs on CPU by default (BLIP-base is
small); this keeps the MPS device free for the diffusion pipeline.
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image


_BLIP_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "blip")
_MODEL_ID = "Salesforce/blip-image-captioning-base"
_cache = {"proc": None, "model": None, "failed": False}


def _ensure_loaded() -> bool:
    if _cache["model"] is not None:
        return True
    if _cache["failed"]:
        return False
    if not os.path.isdir(_BLIP_LOCAL_DIR) or not os.listdir(_BLIP_LOCAL_DIR):
        print("BLIP weights not found locally. Expected directory: " + os.path.abspath(_BLIP_LOCAL_DIR))
        _cache["failed"] = True
        return False
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _cache["proc"] = BlipProcessor.from_pretrained(_BLIP_LOCAL_DIR)
        _cache["model"] = BlipForConditionalGeneration.from_pretrained(_BLIP_LOCAL_DIR).to("cpu")
        return True
    except Exception:
        _cache["failed"] = True
        return False


def caption_image(img: Image.Image, max_new_tokens: int = 40,
                  prompt: Optional[str] = None) -> str:
    """Return a short caption for ``img``.

    Falls back to a neutral placeholder if the caption model is unavailable, so
    the UI never crashes when transformers/BLIP cannot be loaded offline.
    """
    if not _ensure_loaded():
        return "a detailed scene"
    proc = _cache["proc"]
    model = _cache["model"]
    inputs = proc(img.convert("RGB"), text=prompt, return_tensors="pt").to("cpu") \
        if prompt else proc(img.convert("RGB"), return_tensors="pt").to("cpu")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return proc.decode(out[0], skip_special_tokens=True).strip()


def is_available() -> bool:
    """Whether BLIP can be loaded (does not force a download if already failed)."""
    return _ensure_loaded()
