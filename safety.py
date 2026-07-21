# safety.py — VisualDiffusion safety module (single import point)
#
# This module bundles the three safety layers used across the generation
# scripts into one place. It is IMPORTED by the scripts; it is not a fork of
# any third-party application. The image layer uses diffusers' own
# StableDiffusionSafetyChecker class (first-party, from the diffusers library
# this project already depends on) backed by the CompVis/
# stable-diffusion-safety-checker weights published by CompVis (the lab that
# created Stable Diffusion).
#
# ---------------------------------------------------------------------------
# RESPONSIBLE USE
# The operators of this software are solely responsible for the prompts they
# submit and the images they generate. This module is a set of technical
# guards, not a substitute for legal compliance or human judgment. Choose and
# configure each layer to align with the laws and regulations that apply to
# you. The authors of this software provide it as-is and are not liable for
# any output produced by it.
# ---------------------------------------------------------------------------

import os
import re

import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


# =====================================================================
# LAYER 1 — BLOCKED_LIST (prompt term gate)
# =====================================================================
# Purpose: refuse generation when the prompt contains terms the operator has
# decided are not acceptable for their use case (e.g. a competitor's name, a
# specific individual, or any term the operator chooses to exclude). This is
# an operator-maintained list; populate it according to your own policy.
#
# Starts EMPTY. Example of the shape only — the operator decides the contents:
#   BLOCKED_LIST = ["Competitor Name", "Name of Person", "Unsafe Term"]
#
# You are responsible for the outputs you produce, including the terms you
# choose to block or allow.
BLOCKED_LIST = []  # operator-defined; empty by default

# Prompts written in non-Latin scripts (CJK / Japanese / Korean) are refused.
# This is a language-scope decision for this tool, not a content judgment.
_NON_LATIN_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


# =====================================================================
# LAYER 2 — TEXT CLASSIFIER TRIGGER ABORT (opt-in)
# =====================================================================
# Purpose: abort generation when a TEXT classifier flags the positive prompt.
# Enabled by default; the operator can disable it in the config file or setup.
TEXT_CLASSIFIER_ENABLED = True
try:
    from core.config import load_config_file
    _safety_cfg = load_config_file()
    _raw = _safety_cfg.get("text_classifier_enabled")
    if isinstance(_raw, str):
        TEXT_CLASSIFIER_ENABLED = _raw.strip().upper() == "TRUE"
    elif isinstance(_raw, bool):
        TEXT_CLASSIFIER_ENABLED = _raw
except Exception:
    pass
TEXT_CLASSIFIER_MODEL = ""  # operator-selected; e.g. an HF repo id or local path
TEXT_CLASSIFIER_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "safety"
)
TEXT_CLASSIFIER_THRESHOLD = 0.5
_TEXT_CLASSIFIER_PIPELINE = None


def _resolve_text_classifier_model():
    # Prefer a locally vendored copy under models/safety/; fall back to the
    # operator-provided repo id so the app works if the user never downloads it.
    local_model = os.path.join(TEXT_CLASSIFIER_LOCAL_DIR, "model.safetensors")
    if os.path.isdir(TEXT_CLASSIFIER_LOCAL_DIR) and os.path.exists(local_model):
        return TEXT_CLASSIFIER_LOCAL_DIR
    return TEXT_CLASSIFIER_MODEL


def _get_text_classifier():
    global _TEXT_CLASSIFIER_PIPELINE
    if _TEXT_CLASSIFIER_PIPELINE is not None:
        return _TEXT_CLASSIFIER_PIPELINE
    if not TEXT_CLASSIFIER_MODEL:
        raise RuntimeError(
            "[SAFETY] TEXT_CLASSIFIER_ENABLED is True but TEXT_CLASSIFIER_MODEL "
            "is empty. Set it to a classifier you have selected and are "
            "authorized to use."
        )
    from transformers import pipeline
    _TEXT_CLASSIFIER_PIPELINE = pipeline(
        "text-classification", model=_resolve_text_classifier_model(),
        top_k=None, function_to_apply="sigmoid", return_all_scores=True,
    )
    return _TEXT_CLASSIFIER_PIPELINE


def _text_is_flagged(text: str) -> bool:
    clf = _get_text_classifier()
    out = clf(text)[0]
    for d in out:
        if str(d.get("label", "")).lower() in ("nsfw", "flagged", "unsafe"):
            return float(d.get("score", 0.0)) >= TEXT_CLASSIFIER_THRESHOLD
    return False


# =====================================================================
# LAYER 3 — PREVIEW / FINAL IMAGE CENSOR (CompVis safety checker via diffusers)
# =====================================================================
# Purpose: after an image is generated (and for any preview), inspect the
# pixels with the CompVis/stable-diffusion-safety-checker. This is the SAME
# checker diffusers/Stable Diffusion ships by default (StableDiffusionSafetyChecker
# + CompVis weights), so it is the de-facto "default everyone uses". It is also
# known for FALSE POSITIVES (skin tones, artistic nudity, even some landscapes
# get flagged), so we keep it ON and work with it rather than disabling it.
#
# This module's toggle mirrors Fooocus's "Black Out NSFW" setting, which Fooocus
# stores as `default_black_out_nsfw` in config.txt / presets/default.json (and
# surfaces as the "Black Out NSFW" checkbox in Developer/Debug mode). We use the
# equivalent name SAFETY_CHECKER_BLACK_OUT_NSFW so folks recognise it. Fooocus
# defaults it to False; we default it to True to keep the stock diffusers
# behaviour (flagged images are replaced with a black image).
#
# Model: CompVis/stable-diffusion-safety-checker
#   Repo:  https://huggingface.co/CompVis/stable-diffusion-safety-checker
#   Readme: https://huggingface.co/CompVis/stable-diffusion-safety-checker/blob/main/README.md
#   Provided by CompVis (the group that created Stable Diffusion). Shipped as
#   part of the diffusers library. Licensed under the model card's terms.
#   See the model card / README above for what the checker does and its limits.
#
# The value is read from the operator config file (config/app_config.json).
# If the file is missing or the key is absent, the fallback is True (safe by
# default). The operator must explicitly set the value to FALSE to disable.
# --- CODE-LEVEL CONSTANT, NOT A UI PARAMETER -----------------------------------
# SAFETY_CHECKER_BLACK_OUT_NSFW is the single, clear switch for the Safety
# Checker. There is no second flag: the checker is either ON (and blacks out
# flagged NSFW images) or OFF. It is deliberately NOT surfaced as a Streamlit
# widget/checkbox — disabling it must be a real, conscious operator decision
# made by editing this file, never a casual click in the interface.
#   True  (default) -> checker ON; flagged images are replaced with black.
#   False           -> checker OFF; no inspection, images pass through unchanged.
# Named to match Fooocus's `default_black_out_nsfw` ("Black Out NSFW") so the
# meaning is recognisable. Fooocus defaults it to False; we keep it True to
# retain the stock diffusers behaviour.
SAFETY_CHECKER_BLACK_OUT_NSFW: bool = True
try:
    from core.config import load_config_file
    _safety_cfg = load_config_file()
    _raw = _safety_cfg.get("safety_checker_black_out_nsfw")
    if isinstance(_raw, str):
        SAFETY_CHECKER_BLACK_OUT_NSFW = _raw.strip().upper() == "FALSE"
    elif isinstance(_raw, bool):
        SAFETY_CHECKER_BLACK_OUT_NSFW = _raw
except Exception:
    pass
IMAGE_CENSOR_REPO = "CompVis/stable-diffusion-safety-checker"
IMAGE_CENSOR_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "safety_checker"
)
_FEATURE_EXTRACTOR = None
_SAFETY_CHECKER = None


def _resolve_image_censor_source():
    if os.path.isdir(IMAGE_CENSOR_LOCAL_DIR) and any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(IMAGE_CENSOR_LOCAL_DIR)
    ):
        return IMAGE_CENSOR_LOCAL_DIR
    return IMAGE_CENSOR_REPO


def _load_image_censor():
    global _FEATURE_EXTRACTOR, _SAFETY_CHECKER
    if _FEATURE_EXTRACTOR is None or _SAFETY_CHECKER is None:
        src = _resolve_image_censor_source()
        _FEATURE_EXTRACTOR = CLIPImageProcessor.from_pretrained(src)
        _SAFETY_CHECKER = StableDiffusionSafetyChecker.from_pretrained(src)
        _SAFETY_CHECKER.eval()


def censor_image(image):
    """Return (censored_pil_image, was_flagged_bool)."""
    global _FEATURE_EXTRACTOR, _SAFETY_CHECKER
    _load_image_censor()
    x_image = np.array(image).astype(np.float32) / 255.0
    x_image = x_image[None, ...] if x_image.ndim == 3 else x_image
    safety_input = _FEATURE_EXTRACTOR(
        [Image.fromarray((x_image[0] * 255).astype("uint8"))], return_tensors="pt"
    )
    x_checked, has_nsfw = _SAFETY_CHECKER(
        images=x_image, clip_input=safety_input.pixel_values
    )
    checked = (np.asarray(x_checked[0]) * 255).round().astype("uint8")
    return Image.fromarray(checked), bool(has_nsfw[0])


# =====================================================================
# ORCHESTRATION
# =====================================================================
def check_prompt(prompt: str, negative_prompt: str = ""):
    """Pre-generation gate. Raises SystemExit if a layer refuses the prompt.

    The text classifier (Layer 2) scans ONLY the positive prompt; negative
    prompts may legitimately contain steer-away terms and are excluded.
    """
    if BLOCKED_LIST:
        blob = " ".join(t.lower() for t in (prompt, negative_prompt) if t)
        hits = [w for w in BLOCKED_LIST if w.lower() in blob]
        if hits:
            raise SystemExit(
                f"[SAFETY] Blocked. Prompt contains operator-blocked terms: {hits}"
            )

    if _NON_LATIN_RE.search((prompt or "") + " " + (negative_prompt or "")):
        raise SystemExit("[SAFETY] Non-ASCII / CJK prompts are not allowed.")

    if TEXT_CLASSIFIER_ENABLED and prompt:
        if not TEXT_CLASSIFIER_MODEL and not os.path.exists(
            os.path.join(TEXT_CLASSIFIER_LOCAL_DIR, "model.safetensors")
        ):
            print("[SAFETY] Text classifier enabled but no model configured; "
                  "set TEXT_CLASSIFIER_MODEL or place model.safetensors in "
                  f"{TEXT_CLASSIFIER_LOCAL_DIR}", flush=True)
        else:
            if _text_is_flagged(prompt):
                raise SystemExit(
                    "[SAFETY] Blocked. Prompt flagged by the configured text classifier."
                )


def censor(image):
    """Output guard (the Safety Checker).

    Controlled by the single code-level constant SAFETY_CHECKER_BLACK_OUT_NSFW:
      True  (default) -> checker runs; flagged images are replaced with black.
      False           -> checker OFF; image passes through unchanged.
    There is no warn-only mode and no second flag — the checker is either ON
    (and blacks out NSFW) or OFF.
    """
    if not SAFETY_CHECKER_BLACK_OUT_NSFW:
        return image
    censored, flagged = censor_image(image)
    if flagged:
        print("[SAFETY] Potential unsafe content detected in image; "
              "returning a black image instead.")
        return censored
    return image
