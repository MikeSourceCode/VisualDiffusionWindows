"""Visual Diffusion — consolidated Streamlit application (v1).

Single tabbed UI that replaces the collection of standalone generation scripts.
All tabs share one ``AppConfig`` (the sidebar writes it; tabs read it) and the
shared ``core`` generation engine, so settings stay consistent everywhere.

Run:
    development/bin/streamlit run app.py
"""

import json
import os
import random
import signal
import sys
import threading
import time
import warnings

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _quit(*_):
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    os._exit(0)


if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, _quit)

from core import (
    AppConfig,
    Backend,
    VRAMState,
    discover_all,
    detect_backend,
    detect_vram_mb,
    get_vram_state,
    load_base_pipeline,
    load_controlnet_pipeline,
    load_loras_into_unet,
    generate,
    default_vae_path,
    load_init_image_from_path,
    load_tag_groups,
    preprocess,
)
from core.tags import shuffle_tag_groups, _compose_color_noun
from core import persistence
from core.compatibility import validate_local_model_set
from safety import censor_image, SAFETY_CHECKER_BLACK_OUT_NSFW
from ui import render_share_card, render_phone_frame

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*resource_tracker.*leaked semaphore.*",
)

st.set_page_config(page_title="Visual Diffusion Studio", layout="wide")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_CHECKPOINT = "sd_xl_base_1.0.safetensors"  # safe SFW base default
MIN_LORA_WEIGHT = 0.0
MAX_LORA_WEIGHT = 2.0
DEFAULT_LORA_WEIGHT = 0.6

UI_CSS = """
<style>
.vd-canvas { border-radius: 14px; border: 1px solid #2b2f3a; box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.vd-title { background: linear-gradient(90deg,#7c5cff,#21d4fd); -webkit-background-clip: text;
            -webkit-text-fill-color: transparent; font-weight: 800; }
/* Streamlit main container: relax the default padding/width so the layout
   uses the full viewport. */
.block-container {
    padding: 1rem 4rem 1rem;
    max-width: initial;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def cached_load_base_pipeline(model_path, vae_path, architecture):
    backend = detect_backend()
    vram_state = get_vram_state(backend, detect_vram_mb(backend))
    pipe = load_base_pipeline(model_path, vae_path, backend, architecture, img2img=True)
    return pipe, backend, vram_state


def refresh_pipeline(cfg: AppConfig, assets: dict):
    """(Re)build the cached base pipeline when checkpoint/VAE/arch changes."""
    if not cfg.checkpoint:
        return None, None, None
    d = assets["dirs"]

    # Determine whether the selected checkpoint is a single file or a model-set folder.
    if cfg.checkpoint in assets.get("model_sets", {}):
        ckpt_path = os.path.join(d["model_set"], cfg.checkpoint)
    else:
        ckpt_path = os.path.join(d["checkpoints"], cfg.checkpoint)

    vae_path = os.path.join(d["vae"], cfg.vae) if cfg.vae else default_vae_path(d["vae"])
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: {cfg.checkpoint}")
        return None, None, None

    if os.path.isdir(ckpt_path):
        result = validate_local_model_set(ckpt_path, backend=cfg.architecture)
        if not result:
            st.error(
                f"**'{cfg.checkpoint}' is not compatible.**\\n\\n" +
                "\\n".join(f"- {r}" for r in result.reasons) +
                "\\n\\nRe-download it with:\\n" +
                f"```bash\\npython getmodel.py {cfg.checkpoint.replace('_', '/')}\\n```"
            )
            return None, None, None

    cache_key = (cfg.checkpoint, cfg.vae, cfg.architecture, os.path.isdir(ckpt_path))
    is_new = st.session_state.get("vd_cache_key") != cache_key
    if is_new:
        cached_load_base_pipeline.clear()
        st.session_state.vd_cache_key = cache_key
    if is_new:
        label = f"'{cfg.checkpoint}' (folder)" if os.path.isdir(ckpt_path) else f"'{cfg.checkpoint}'"
        print(f"[load] building pipeline: ckpt={cfg.checkpoint} vae={vae_path or 'fp16-fix'} arch={cfg.architecture}", flush=True)
        with st.spinner(f"Loading {label} into memory — first load is slow..."):
            t0 = time.time()
            pipe, be, vr = cached_load_base_pipeline(ckpt_path, vae_path, cfg.architecture)
            print(f"[load] pipeline ready in {time.time() - t0:.1f}s", flush=True)
            return pipe, be, vr
    return cached_load_base_pipeline(ckpt_path, vae_path, cfg.architecture)


def universal_sidebar(assets: dict) -> AppConfig:
    """Render the sidebar and write every widget value into ``app_config``."""
    cfg: AppConfig = st.session_state.app_config

    st.sidebar.markdown("### 🎛️ Global Settings")

    ckpt_names = list(assets["checkpoints"].keys())
    model_set_names = list(assets.get("model_sets", {}).keys())
    all_ckpt_names = ckpt_names + [f"📁 {n}" for n in model_set_names]

    def _display_name(name: str) -> str:
        if name.startswith("📁 "):
            return name
        if name in assets.get("model_sets", {}):
            return f"📁 {name}"
        return name

    def _resolve_name(name: str) -> tuple[str, str]:
        """Return (display_name, storage_name) where storage_name is the raw name."""
        if name.startswith("📁 "):
            return name, name[2:]
        return name, name

    current_display = _display_name(cfg.checkpoint)
    if current_display not in all_ckpt_names:
        cfg.checkpoint = DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT in all_ckpt_names else (all_ckpt_names[0] if all_ckpt_names else "")
    current_display = _display_name(cfg.checkpoint)
    ckpt_idx = all_ckpt_names.index(current_display) if current_display in all_ckpt_names else 0
    sel_display = st.sidebar.selectbox("Checkpoint", all_ckpt_names, index=ckpt_idx)
    sel_storage, sel_name = _resolve_name(sel_display)

    ckpt_meta = assets["checkpoints"].get(sel_name) or assets.get("model_sets", {}).get(sel_name) or {}
    cfg.architecture = ckpt_meta.get("arch", cfg.architecture)
    cfg.checkpoint = sel_name
    st.sidebar.caption(f"Architecture: **{cfg.architecture}**")

    vae_opts = [""] + assets["vaes"]
    vae_labels = ["Built-in (default)"] + assets["vaes"]
    if cfg.vae not in vae_opts:
        cfg.vae = ""
    vae_idx = vae_opts.index(cfg.vae)
    selected_label = st.sidebar.selectbox("Custom VAE", vae_labels, index=vae_idx)
    cfg.vae = vae_opts[vae_labels.index(selected_label)]

    # Up to two stackable LoRAs.
    lora_opts = [""] + assets["loras"]
    cur = cfg.loras if cfg.loras else [("", 0.0), ("", 0.0)]
    while len(cur) < 2:
        cur.append(("", 0.0))

    l1_name = st.sidebar.selectbox("LoRA 1", lora_opts, index=_idx(lora_opts, cur[0][0]))
    l1_w = st.sidebar.slider("LoRA 1 Weight", MIN_LORA_WEIGHT, MAX_LORA_WEIGHT, float(cur[0][1] or DEFAULT_LORA_WEIGHT), 0.05)
    l2_name = st.sidebar.selectbox("LoRA 2", lora_opts, index=_idx(lora_opts, cur[1][0]))
    l2_w = st.sidebar.slider("LoRA 2 Weight", MIN_LORA_WEIGHT, MAX_LORA_WEIGHT, float(cur[1][1] or DEFAULT_LORA_WEIGHT), 0.05)
    # A blank selection means "no LoRA" regardless of the slider position.
    cfg.loras = [(l1_name, l1_w if l1_name else 0.0), (l2_name, l2_w if l2_name else 0.0)]

    st.sidebar.divider()
    cfg.steps = st.sidebar.slider("Inference Steps", 8, 60, cfg.steps)
    cfg.cfg_scale = st.sidebar.slider("Guidance Scale (CFG)", 1.0, 15.0, cfg.cfg_scale, 0.5)
    cfg.seed = st.sidebar.number_input("Seed (0 = random)", min_value=0, value=cfg.seed, step=1)
    cfg.strength = st.sidebar.slider("Img2Img Strength", 0.0, 1.0, cfg.strength, 0.05)
    cfg.sharpness = st.sidebar.slider("Sharpness", 0.0, 2.0, cfg.sharpness, 0.1)

    st.sidebar.divider()
    # Output aspect ratio for text-to-image (img2img uses the source size).
    ratios = {
        "9:16 Portrait (384x688)": (384, 688),
        "9:16 Portrait (768x1368)": (768, 1368),
        "3:4 Portrait (896x1152)": (896, 1152),
        "1:1 Square (1024x1024)": (1024, 1024),
        "4:3 Landscape (1152x896)": (1152, 896),
        "16:9 Landscape (1472x832)": (1472, 832),
    }
    ratio_labels = list(ratios.keys())
    cur = (cfg.width, cfg.height)
    cur_label = next((k for k, v in ratios.items() if v == cur), ratio_labels[0])
    chosen = st.sidebar.selectbox("Output Size (Text→Image)", ratio_labels,
                                  index=ratio_labels.index(cur_label))
    cfg.width, cfg.height = ratios[chosen]

    st.sidebar.divider()
    cfg.show_previews = st.sidebar.checkbox("Show Live Previews", cfg.show_previews)
    cfg.preview_frequency = st.sidebar.slider("Preview every N steps", 1, 10, cfg.preview_frequency,
                                               disabled=not cfg.show_previews)

    return cfg


def _idx(options, value):
    return options.index(value) if value in options else 0


@st.cache_data(show_spinner=False)
def _tag_groups():
    groups = load_tag_groups()
    # Deterministic shuffle per session so the palette isn't static.
    if "vd_tag_shuffle_seed" not in st.session_state:
        st.session_state.vd_tag_shuffle_seed = random.randint(0, 2**31)
    return shuffle_tag_groups(groups, st.session_state.vd_tag_shuffle_seed)


def _clear_tag_keys(keys):
    """on_click handler: reset each pill widget's selection before re-render."""
    for k in keys:
        st.session_state[k] = []


def _flat_tag_map():
    """Return {display_label: prompt_text} across all tag groups."""
    label_to_text = {}
    for group in _tag_groups():
        for t in group["tags"]:
            label = f"{t['emoji']} {t['label']}".strip()
            label_to_text[label] = t["text"]
    return label_to_text


def _sync_tags_into_prompt(prompt_key, pills_key, applied_key, label_to_text):
    """on_change handler for the pills: reconcile selected tags into the prompt.

    Appends the text of newly selected pills and removes the text of newly
    deselected pills, preserving the user's manual edits. Pills stay selected.
    """
    selected_labels = st.session_state.get(pills_key) or []
    selected_texts = [label_to_text[l] for l in selected_labels if l in label_to_text]
    previously = st.session_state.get(applied_key, [])

    prompt = st.session_state.get(prompt_key, "") or ""

    # Remove texts for tags that were just deselected.
    for text in previously:
        if text not in selected_texts:
            prompt = _remove_fragment(prompt, text)

    # Append texts for tags that were just selected.
    for text in selected_texts:
        if text not in previously and not _contains_fragment(prompt, text):
            prompt = _append_fragment(prompt, text)

    st.session_state[prompt_key] = prompt
    st.session_state[applied_key] = selected_texts


def _compose_color_into_prompt(prompt_key, color_pills_key, noun_applied_key, label_to_text):
    """Compose selected color(s) with the last selected noun and inject into prompt.

    The color selection is cleared after composing so it doesn't re-apply to
    the next noun unintentionally. The noun selection is kept.
    """
    color_labels = st.session_state.get(color_pills_key) or []
    color_texts = [label_to_text[l] for l in color_labels if l in label_to_text]
    noun_texts = st.session_state.get(noun_applied_key, [])

    prompt = st.session_state.get(prompt_key, "") or ""

    # Remove any previous color composition from the prompt.
    for ct in color_texts:
        for nt in noun_texts:
            composed = _compose_color_noun(ct, nt)
            prompt = _remove_fragment(prompt, composed)

    # Compose the most recent noun with each selected color.
    if noun_texts and color_texts:
        noun = noun_texts[-1]
        for color in color_texts:
            composed = _compose_color_noun(color, noun)
            if composed and not _contains_fragment(prompt, composed):
                prompt = _append_fragment(prompt, composed)

    st.session_state[prompt_key] = prompt
    # Reset color selection so it doesn't double-apply.
    st.session_state[color_pills_key] = []


def _contains_fragment(prompt, fragment):
    parts = [p.strip() for p in (prompt or "").split(",")]
    return fragment.strip() in parts


def _append_fragment(prompt, fragment):
    prompt = (prompt or "").strip().rstrip(",").strip()
    return f"{prompt}, {fragment}" if prompt else fragment


def _remove_fragment(prompt, fragment):
    parts = [p.strip() for p in (prompt or "").split(",")]
    parts = [p for p in parts if p and p != fragment.strip()]
    return ", ".join(parts)


def prompt_with_tags(key_prefix, default_prompt="", default_negative="",
                     positive_label="Positive Prompt", negative_label="Negative Prompt",
                     cfg=None, tab=None, show_tags: bool = True):
    """Render, in order: emoji pills -> Positive Prompt -> Negative Prompt.

    If ``show_tags`` is False, the emoji/color/surprise/clear-tag UI is hidden
    but the prompt text boxes still render and return their values.

    Defaults come from the config file in this order:
      1. ``cfg.tab_config(tab)`` (per-tab overrides)
      2. ``cfg.default_prompt`` / ``cfg.default_negative_prompt`` (common)
      3. explicit function arguments
      4. empty string
    """
    # Resolve defaults from config (tab > common > explicit > empty).
    tab_cfg = {}
    if cfg is not None and tab:
        tab_cfg = cfg.tab_config(tab) or {}
    if not default_prompt:
        default_prompt = tab_cfg.get("default_prompt") or (cfg.default_prompt if cfg else "") or ""
    if not default_negative:
        default_negative = tab_cfg.get("default_negative_prompt") or (cfg.default_negative_prompt if cfg else "") or ""

    groups = _tag_groups()
    regular_groups = [g for g in groups if not g.get("composable")]
    composable_groups = [g for g in groups if g.get("composable")]

    prompt_key = f"{key_prefix}_prompt"
    neg_key = f"{key_prefix}_neg"
    pills_key = f"{key_prefix}_tags_flat"
    applied_key = f"{key_prefix}_applied_tags"

    # Seed the prompt box once so we can safely mutate it from the pill callback.
    st.session_state.setdefault(prompt_key, default_prompt)

    # 1) Regular emoji pills first (right under the tab heading).
    regular_label_to_text = {}
    for group in regular_groups:
        for t in group["tags"]:
            label = f"{t['emoji']} {t['label']}".strip()
            regular_label_to_text[label] = t["text"]

    if show_tags and regular_label_to_text:
        st.caption("Click emoji tags to add them to your prompt (click again to remove)")
        st.pills(
            "Tags",
            list(regular_label_to_text.keys()),
            selection_mode="multi",
            key=pills_key,
            label_visibility="collapsed",
            on_change=_sync_tags_into_prompt,
            args=(prompt_key, pills_key, applied_key, regular_label_to_text),
        )
        if st.session_state.get(pills_key):
            st.button("🧹 Clear tags", key=f"{key_prefix}_clear_tags",
                      on_click=_clear_tag_keys, args=([pills_key],))

    # 1b) Composable color pills (compose with the last selected noun).
    if show_tags:
        for cgroup in composable_groups:
            cname = cgroup["name"]
            ctags = cgroup["tags"]
            color_labels = [f"{t['emoji']} {t['label']}".strip() for t in ctags]
            color_label_to_text = {f"{t['emoji']} {t['label']}".strip(): t["text"] for t in ctags}
            color_pills_key = f"{key_prefix}_color_tags"
            color_applied_key = f"{key_prefix}_color_applied"

            st.caption(f"Pick a color to combine with your subject (e.g. blue + car = a blue car)")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.pills(
                    cname,
                    color_labels,
                    selection_mode="multi",
                    key=color_pills_key,
                    label_visibility="collapsed",
                    on_change=_compose_color_into_prompt,
                    args=(prompt_key, color_pills_key, applied_key, color_label_to_text),
                )
            with c2:
                subject_texts = [t["text"] for g in regular_groups for t in g.get("tags", []) if g.get("name") == "Subject"]

                def _surprise_color(key_prefix=key_prefix, color_pills_key=color_pills_key,
                                      applied_key=applied_key, color_label_to_text=color_label_to_text,
                                      subject_texts=subject_texts):
                    import random as _rnd
                    if not color_label_to_text:
                        st.warning("No color tags available.")
                        return
                    color = _rnd.choice(list(color_label_to_text.values()))
                    applied = st.session_state.get(applied_key, [])
                    subject_candidates = [t for t in applied if t in subject_texts]
                    if subject_candidates:
                        noun = subject_candidates[-1]
                    elif subject_texts:
                        noun = _rnd.choice(subject_texts)
                    else:
                        st.warning("No subject tags defined — add some in data/tags.json.")
                        return
                    composed = _compose_color_noun(color, noun)
                    prompt = st.session_state.get(f"{key_prefix}_prompt", "") or ""
                    if composed and not _contains_fragment(prompt, composed):
                        prompt = _append_fragment(prompt, composed)
                    st.session_state[f"{key_prefix}_prompt"] = prompt
                    st.session_state[color_pills_key] = []
                    st.rerun()

                st.button("🎲 Surprise", key=f"{key_prefix}_surprise", on_click=_surprise_color)

    # 2) Positive prompt (tags are injected into this box).
    positive = st.text_area(positive_label, height=120, key=prompt_key)
    # 3) Negative prompt directly below.
    negative = st.text_area(negative_label, value=default_negative, height=90, key=neg_key)
    return positive, negative


def preset_panel():
    """Sidebar preset save/load via SQLite."""
    st.sidebar.divider()
    st.sidebar.markdown("### 💾 Presets")
    presets = persistence.list_presets()
    names = [p["name"] for p in presets]
    if names:
        chosen = st.sidebar.selectbox("Load preset", [""] + names)
        if st.sidebar.button("Load", disabled=not chosen):
            loaded = persistence.load_preset(chosen)
            if loaded:
                st.session_state.app_config = loaded
                st.rerun()
    pname = st.sidebar.text_input("Save preset as", placeholder="my-favorite-config")
    if st.sidebar.button("Save current", disabled=not pname.strip()):
        persistence.save_preset(pname.strip(), st.session_state.app_config)
        st.sidebar.success(f"Saved '{pname.strip()}'")
        st.rerun()


# --------------------------------------------------------------------------- #
# Feature tabs
# --------------------------------------------------------------------------- #
def tab_image_to_image(assets):
    st.markdown("### Image → Image")
    st.caption("Generate from a source image + prompt. Uses global sidebar settings.")
    cfg = st.session_state.app_config
    pipe, backend, vram_state = refresh_pipeline(cfg, assets)
    if pipe is None:
        st.warning("No checkpoint selected.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Source image", type=["png", "jpg", "jpeg", "webp"])
        prompt, negative = prompt_with_tags("i2i", cfg=cfg, tab="image_to_image", show_tags=False)
        gen = st.button("⚡ Generate", use_container_width=True, key="i2i_gen")
    with col2:
        out = st.empty()

    if uploaded is not None:
        init_img = _pil_from_upload(uploaded)

    if gen and uploaded is None:
        st.warning("Upload a source image first, or use the **Text → Image** tab to generate without one.")

    if gen and uploaded is not None:
        init_img = _pil_from_upload(uploaded)
        render_phone_frame(init_img, caption="Source", target=out)
        _run_generation(pipe, backend, vram_state, cfg, prompt, negative,
                        init_image=init_img, preview_slot=out,
                        share_card=True, share_text=prompt, tab="Image → Image", key_prefix="share_i2i")
    else:
        if uploaded is not None:
            render_phone_frame(init_img, caption="Source", target=out)
        else:
            cached = st.session_state.get("vd_result_Image → Image")
            if cached is not None:
                render_share_card(cached["result"], caption="9:16",
                                  share_text=cached.get("share_text", ""),
                                  size=cached.get("size"), target=out, key_prefix="share_i2i")
            else:
                render_phone_frame(None, caption="9:16", target=out)


def tab_image_control(assets):
    st.markdown("### Image Control → Image")
    st.caption("Structure-guided generation via ControlNet using PyraCanny (pyramid multi-scale edges).")
    cfg = st.session_state.app_config
    controlnets = assets["controlnets"]
    if not controlnets and not _controlnet_dirs(assets):
        st.info("No ControlNet found in models/controlnet/. Download e.g. controlnet-canny-sdxl-1.0 to enable this tab.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        prompt, negative = prompt_with_tags(
            "ic", cfg=cfg, tab="image_control")
        uploaded = st.file_uploader("Source image", type=["png", "jpg", "jpeg"], key="ic_src")
        method = st.selectbox("Edge method", ["PyraCanny", "Canny"], key="ic_method")
        cA, cB = st.columns(2)
        with cA:
            low = st.slider("Low threshold", 0, 255, 64 if method == "PyraCanny" else 100, key="ic_low")
        with cB:
            high = st.slider("High threshold", 0, 255, 128 if method == "PyraCanny" else 200, key="ic_high")
        levels = st.slider("Pyramid levels", 1, 5, 3, key="ic_levels",
                           disabled=(method != "PyraCanny"))
        cfg.controlnet_strength = st.slider("Control strength", 0.0, 1.0, cfg.controlnet_strength, 0.05, key="ic_cs")
        gen = st.button("⚡ Generate", use_container_width=True, key="ic_gen")
    with col2:
        map_slot = st.empty()
        out = st.empty()

    # Live control-map preview so the user can tune thresholds before generating.
    control_img = None
    if uploaded is not None:
        init_img = _pil_from_upload(uploaded)
        control_img = _make_control_map(init_img, method, low, high, levels)
        if control_img is not None:
            map_slot.image(control_img, caption=f"{method} control map", width=320)

    if gen and uploaded is not None and control_img is not None:
        pipe, backend, vram_state = _refresh_controlnet_pipeline(cfg, assets, prefer="canny")
        if pipe is None:
            return
        init_img = _pil_from_upload(uploaded)
        _run_generation(pipe, backend, vram_state, cfg, prompt, negative,
                        init_image=init_img, control_image=control_img,
                        preview_slot=out, share_card=True, share_text=prompt, tab="Image Control", key_prefix="share_ic")
    else:
        cached = st.session_state.get("vd_result_Image Control")
        if cached is not None:
            render_share_card(cached["result"], caption="9:16",
                              share_text=cached.get("share_text", ""),
                              size=cached.get("size"), target=out, key_prefix="share_ic")
        else:
            render_phone_frame(None, caption="9:16", target=out)


def tab_ip_consistency(assets):
    st.markdown("### IP Consistency")
    st.caption("Preserve character likeness across scene changes via IP-Adapter.")
    cfg = st.session_state.app_config
    pipe, backend, vram_state = refresh_pipeline(cfg, assets)
    if pipe is None:
        st.warning("No checkpoint selected.")
        return
    st.info("IP-Adapter integration is wired to the same engine; the IP-Adapter image encoder "
            "module loads on demand. (See img_ip_adapter_test.py for the working pipeline.)")
    col1, col2 = st.columns([1, 1])
    with col1:
        ref = st.file_uploader("Reference image (character)", type=["png", "jpg", "jpeg"], key="ip_ref")
        prompt, negative = prompt_with_tags("ip", cfg=cfg, tab="ip_consistency")
        cfg.ip_adapter_scale = st.slider("IP scale", 0.0, 1.0, cfg.ip_adapter_scale, 0.05, key="ip_scale")
        gen = st.button("⚡ Generate", use_container_width=True, key="ip_gen")
    with col2:
        out = st.empty()
    if gen and ref is not None:
        ref_img = _pil_from_upload(ref)
        # Show the reference image INSIDE the phone frame before denoising.
        render_phone_frame(ref_img, caption="Reference", target=out)
        _run_generation(pipe, backend, vram_state, cfg, prompt, negative,
                        init_image=ref_img, preview_slot=out, tab="IP Consistency")


def tab_text_to_image(assets):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Text → Image")
        st.caption("Pure text-to-image. No upload required — pick tags or type a prompt, then Generate.")
        cfg = st.session_state.app_config
        pipe, backend, vram_state = refresh_pipeline(cfg, assets)
        if pipe is None:
            st.warning("No checkpoint selected.")
            return
        prompt, negative = prompt_with_tags("t2i", cfg=cfg, tab="text_to_image")
        gen = st.button("⚡ Generate", use_container_width=True, key="t2i_gen")
    with col2:
        out = st.empty()

    if gen and prompt.strip():
        _run_generation(pipe, backend, vram_state, cfg, prompt, negative,
                        init_image=None, preview_slot=out,
                        share_card=True, share_text=prompt, tab="Text → Image", key_prefix="share_t2i")
    elif gen:
        st.warning("Enter a prompt or pick some tags first.")
    else:
        cached = st.session_state.get("vd_result_Text → Image")
        if cached is not None:
            render_share_card(cached["result"], caption="9:16",
                              share_text=cached.get("share_text", ""),
                              size=cached.get("size"), target=out, key_prefix="share_t2i")
        else:
            render_phone_frame(None, caption="9:16", target=out)


def tab_pixel_art(assets):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Text → Pixel Art")
        st.caption("Generate an image, then pixelate it with nearest-neighbor downscaling.")
        cfg = st.session_state.app_config
        pipe, backend, vram_state = refresh_pipeline(cfg, assets)
        if pipe is None:
            st.warning("No checkpoint selected.")
            return
        prompt, negative = prompt_with_tags(
            "px",
            cfg=cfg,
            tab="pixel_art",
            default_prompt="pixel art, 16-bit SNES style, limited color palette, clean pixels, dithering, retro game sprite, transparent background",
            default_negative="3d, realistic, blurry, smooth, anti-aliased, photorealistic, soft edges, gradient",
        )
        pixel_scale = st.slider("Pixel Scale", 2, 32, 8, 1,
                                help=" Divisor for nearest-neighbor downscale. Higher = chunkier pixels.")
        gen = st.button("🎮 Generate Pixel Art", use_container_width=True, key="px_gen")
    with col2:
        out = st.empty()

    if gen and prompt.strip():
        def _final(shell, result, share_text, cfg, key_prefix):
            from PIL import Image
            w, h = result.size
            small_w, small_h = max(1, w // pixel_scale), max(1, h // pixel_scale)
            small = result.resize((small_w, small_h), Image.NEAREST)
            pixelated = small.resize((w, h), Image.NEAREST)
            st.session_state["vd_result_Text → Pixel Art"] = {
                "result": pixelated,
                "share_text": prompt,
                "size": (w, h),
            }
            with shell.container():
                render_share_card(pixelated, caption=f"{small_w}x{small_h} pixels",
                                  share_text=prompt, size=(w, h),
                                  key_prefix="share_px")

        _run_generation(pipe, backend, vram_state, cfg, prompt, negative,
                        init_image=None, preview_slot=out,
                        share_card=True, share_text=prompt,
                        tab="Text → Pixel Art", key_prefix="share_px_tmp",
                        final_render_callback=_final)
    elif gen:
        st.warning("Enter a prompt or pick some tags first.")
    else:
        cached = st.session_state.get("vd_result_Text → Pixel Art")
        if cached is not None:
            render_share_card(cached["result"], caption="Pixel Art",
                              share_text=cached.get("share_text", ""),
                              size=cached.get("size"), target=out, key_prefix="share_px")
        else:
            render_phone_frame(None, caption="Pixel Art", target=out)


def tab_image_to_text(assets):
    st.markdown("### Image → Text → Image (Brainstorm)")
    st.caption("Describe an uploaded image, add concept tags, then brainstorm a brand-new image "
               "from the text alone. The output draws no pixels with the source.")
    cfg = st.session_state.app_config

    if st.session_state.get("i2t_nsfw_error"):
        st.error("[SAFETY] Potential unsafe content detected in the source image; "
                 "aborting before analysis. Discard the image and upload a different one.")
        st.session_state.pop("i2t_nsfw_error", None)

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Image to analyze", type=["png", "jpg", "jpeg"], key="i2t_src")
        src = None
        if uploaded is not None:
            src = _pil_from_upload(uploaded)
            st.image(src, caption="Source (not used in output)", width=260)

            # Safety gate: refuse NSFW source images before analysis/captioning.
            if SAFETY_CHECKER_BLACK_OUT_NSFW:
                _censored, _flagged = censor_image(src)
                if _flagged:
                    st.session_state["i2t_src"] = None
                    st.session_state.pop("i2t_sig", None)
                    st.session_state.pop("i2t_prompt", None)
                    st.session_state.pop("i2t_applied_tags", None)
                    st.session_state["i2t_nsfw_error"] = True
                    st.rerun()
                    st.stop()

            # Automatically describe the image once per uploaded file, writing
            # the caption into the positive-prompt box shared with prompt_with_tags.
            file_sig = getattr(uploaded, "name", "") + str(getattr(uploaded, "size", ""))
            if st.session_state.get("i2t_sig") != file_sig:
                with st.spinner("🔎 Analyzing image with BLIP — writing a description..."):
                    st.session_state.i2t_prompt = _caption(src)
                st.session_state.i2t_sig = file_sig
                st.session_state.i2t_applied_tags = []  # reset tag sync baseline
                st.rerun()
            else:
                st.success("✅ Auto-described. Edit the prompt below or add tags, then Generate.")

        if st.button("🔁 Re-describe image", use_container_width=True,
                     key="i2t_describe", disabled=src is None):
            with st.spinner("🔎 Analyzing image with BLIP..."):
                st.session_state.i2t_prompt = _caption(src)
                st.session_state.i2t_applied_tags = []
            st.rerun()

        # Emoji pills -> Positive Prompt (auto-filled by BLIP) -> Negative Prompt.
        prompt, negative = prompt_with_tags(
            "i2t",
            positive_label="Description / Positive Prompt (auto-filled from the image — editable)",
            cfg=cfg,
            tab="image_to_text",
            show_tags=False,
        )
        gen = st.button("⚡ Generate new image", use_container_width=True, key="i2t_gen")
    with col2:
        out = st.empty()

    if not gen:
        if uploaded is not None:
            render_phone_frame(src, caption="Source", target=out)
        else:
            cached = st.session_state.get("vd_result_Image → Text")
            if cached is not None:
                render_share_card(cached["result"], caption="9:16",
                                  share_text=cached.get("share_text", ""),
                                  size=cached.get("size"), target=out, key_prefix="share_i2t")
            else:
                render_phone_frame(None, caption="9:16", target=out)

    if gen:
        if not prompt.strip():
            st.warning("Upload an image to auto-describe, type a prompt, and/or pick tags first.")
            return
        pipe, backend, vram_state = refresh_pipeline(cfg, assets)
        if pipe is None:
            st.warning("No checkpoint selected.")
            return
        # Pure txt2img (no init_image) = clean wash.
        _run_generation(pipe, backend, vram_state, cfg, prompt, negative,
                        init_image=None, preview_slot=out,
                        share_card=True, share_text=prompt, tab="Image → Text", key_prefix="share_i2t")


def tab_pose_control(assets):
    st.markdown("### Pose Control")
    st.caption("Structure control via OpenPose. Upload a reference image (content) and a pose source "
               "(image or JSON keypoints) to apply the pose to the reference.")
    cfg = st.session_state.app_config
    if not _controlnet_dirs(assets):
        st.info("No ControlNet found in models/controlnet/. Download an OpenPose SDXL ControlNet to enable this tab.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        mode = st.radio("Pose source", ["From image", "From JSON keypoints"], key="pose_mode", horizontal=True)
        pose_img = None
        if mode == "From image":
            uploaded = st.file_uploader("Pose reference image", type=["png", "jpg", "jpeg"], key="pose_src")
            if uploaded is not None:
                src = _pil_from_upload(uploaded)
                with st.spinner("Detecting pose..."):
                    pose_img = _detect_pose(src)
                if pose_img is None:
                    st.error("OpenPose detection failed (annotator weights may be unavailable offline).")
        else:
            json_file = st.file_uploader("OpenPose JSON", type=["json"], key="pose_json")
            jc1, jc2 = st.columns(2)
            with jc1:
                w = st.number_input("Canvas width", 256, 2048, 832, 8, key="pose_w")
            with jc2:
                h = st.number_input("Canvas height", 256, 2048, 1472, 8, key="pose_h")
            if json_file is not None:
                pose_img = _skeleton_from_json(json_file.getvalue(), int(w), int(h))

        ref_img = None
        ref_file = st.file_uploader("Reference image (content to apply pose to)", type=["png", "jpg", "jpeg"], key="pose_ref")
        if ref_file is not None:
            ref_img = _pil_from_upload(ref_file)

        prompt, negative = prompt_with_tags(
            "pose",
            cfg=cfg,
            tab="pose_control",
        )
        cfg.controlnet_strength = st.slider("Pose strength", 0.0, 1.0, cfg.controlnet_strength, 0.05, key="pose_cs")
        gen = st.button("⚡ Generate", use_container_width=True, key="pose_gen", disabled=pose_img is None)
    with col2:
        out = st.empty()

    if gen and pose_img is not None:
        if ref_img is None:
            st.warning("Upload a reference image to apply the pose to, or the output will be uncontrolled.")
        pipe, backend, vram_state = _refresh_controlnet_pipeline(cfg, assets, prefer="pose")
        if pipe is None:
            return
        _run_generation(pipe, backend, vram_state, cfg, prompt, negative,
                        init_image=ref_img, control_image=pose_img,
                        preview_slot=out, share_card=True, share_text=prompt, tab="Pose Control", key_prefix="share_pose")
    else:
        cached = st.session_state.get("vd_result_Pose Control")
        if cached is not None:
            render_share_card(cached["result"], caption="9:16",
                              share_text=cached.get("share_text", ""),
                              size=cached.get("size"), target=out, key_prefix="share_pose")
        else:
            render_phone_frame(None, caption="9:16", target=out)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pil_from_upload(uploaded):
    from PIL import Image
    return Image.open(uploaded).convert("RGB")


def _image_to_truecolor_ascii(img, columns: int = 80) -> str:
    """Render a PIL image as truecolor ASCII (port of image_to_image_diffusion.py).

    Prints to the terminal during generation so previews are visible in the
    console as well as inside the phone shell. Uses ANSI truecolor escapes with
    a luminance ramp; no extra dependency beyond Pillow.
    """
    ramp = " .:-=+*#%@"
    w, h = img.size
    rows = max(1, int(columns * (h / w) * 0.55))
    small = img.convert("RGB").resize((columns, rows))
    px = small.load()
    n = len(ramp)
    out = []
    for y in range(rows):
        cells = []
        for x in range(columns):
            r, g, b = px[x, y]
            lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
            ch = ramp[min(n - 1, int(lum * n))]
            cells.append(f"\x1b[38;2;{r};{g};{b}m{ch}")
        cells.append("\x1b[0m")
        out.append("".join(cells))
    return "\n".join(out)


def _print_ascii_preview(img, step: int, columns: int = 80) -> None:
    try:
        print(f"\n--- preview step {step} ---")
        print(_image_to_truecolor_ascii(img, columns))
    except Exception as e:
        print(f"[ASCII] preview step {step} failed: {e}")


def print_telemetry(tab: str, backend, vram_state, architecture: str, checkpoint: str,
                    vae_name: str, loras, steps: int, cfg_scale: float, seed,
                    gen_time: float, width: int, height: int, sharpness: float) -> None:
    """Print a run-summary block to the terminal (ported from the standalone scripts).

    Adds ``tab`` so runs from the consolidated app can be told apart and compared
    against the standalone image_to_image_diffusion.py timings.
    """
    print("\n" + "=" * 80)
    print(f"Tab: {tab} | Model: {checkpoint} | Arch: {architecture}")
    print(f"Backend: {getattr(backend, 'value', backend)} | VRAM: {getattr(vram_state, 'name', vram_state)}")
    print(f"VAE: {vae_name} | Steps: {steps} | CFG: {cfg_scale} | Seed: {seed}")
    print(f"LoRAs: {loras}")
    print(f"Output: {width}x{height} | Sharpness: {sharpness} | Time: {gen_time:.2f}s")
    print("=" * 80)


def _run_generation(pipe, backend, vram_state, cfg, prompt, negative, preview_slot,
                    init_image=None, control_image=None, share_card=False, share_text="",
                    tab="Unknown", key_prefix="share", final_render_callback=None):
    cache_key = f"vd_result_{tab}"
    st.session_state.pop(cache_key, None)
    t_click = time.time()
    t_pre = time.time()
    lora_dir = assets_dirs()["lora"]
    lora_specs = cfg.lora_specs()
    lora_key = tuple(lora_specs)
    cached_specs = st.session_state.get("vd_lora_cache_key")
    if cached_specs != lora_key:
        if lora_specs:
            load_loras_into_unet(pipe, lora_specs, lora_dir, backend)
        else:
            pipe.unload_lora_weights()
        st.session_state.vd_lora_cache_key = lora_key
    t_lora = time.time() - t_pre
    t_to_generate = time.time() - t_click
    print(f"[TIMING] click_to_generate={t_to_generate:.3f}s lora_load={t_lora:.3f}s specs={lora_specs} cached={cached_specs == lora_key}", flush=True)
    print(f"\n[GENERATE] tab={tab} steps={cfg.steps} cfg={cfg.cfg_scale} "
          f"{cfg.width}x{cfg.height} img2img={init_image is not None}", flush=True)

    # Use the st.empty() placeholder directly (canonical in-place replace
    # pattern) so repeated previews REPLACE the frame instead of stacking. The
    # frame is always visible: placeholder -> live previews -> final card, and
    # previews render INSIDE the shell rather than as a bare st.image below it.
    shell = preview_slot
    shell.empty()
    if share_card:
        render_phone_frame(None, caption="9:16", target=shell)
    else:
        shell.caption("Preview will appear here")

    def cb(img, step):
        if cfg.show_previews and (step % cfg.preview_every(cfg.steps) == 0 or step == cfg.steps - 1):
            # Also emit the preview to the terminal as truecolor ASCII (like the
            # standalone scripts) so progress is visible in the console too.
            _print_ascii_preview(img, step)
            shell.empty()
            if share_card:
                render_phone_frame(img, caption=f"Denoised Preview (Step {step})", target=shell)
            else:
                shell.image(img, caption=f"Denoised Preview (Step {step})")

    try:
        t0 = time.time()
        result = generate(pipe, backend, vram_state, cfg, prompt, negative,
                          init_image=init_image, control_image=control_image,
                          preview_callback=cb, lora_dir=lora_dir)
        gen_time = time.time() - t0
        print(f"[TIMING] generate={gen_time:.3f}s", flush=True)
        saved_path = _save_output(result)
        _print_ascii_preview(result, cfg.steps - 1)
        seed = (result.info or {}).get("seed", "na")
        vae_name = cfg.vae or "built-in SDXL"
        print_telemetry(tab, backend, vram_state, cfg.architecture, cfg.checkpoint,
                        vae_name, cfg.lora_specs(), cfg.steps, cfg.cfg_scale, seed,
                        gen_time, cfg.width, cfg.height, cfg.sharpness)
        shell.empty()
        if share_card:
            if final_render_callback is not None:
                final_render_callback(shell, result, share_text, cfg, key_prefix)
            else:
                with shell.container():
                    render_share_card(result, caption="9:16", share_text=share_text,
                                      size=(cfg.width, cfg.height), key_prefix=key_prefix)
            # if saved_path:
            #     shell.caption(f"Saved to {saved_path}")
        else:
            shell.image(result, caption="Final Rendered Output", width="stretch")
            # if saved_path:
            #     shell.caption(f"Saved to {saved_path}")
        if final_render_callback is None:
            st.session_state[f"vd_result_{tab}"] = {
                "result": result,
                "share_text": share_text,
                "size": (cfg.width, cfg.height),
            }
        return result
    except SystemExit as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Generation failed: {e}")
    return None


def _save_output(img):
    """Auto-save a generated image to ./outputs with a timestamp+seed filename."""
    import time
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    seed = (img.info or {}).get("seed", "na")
    fname = f"visualdiffusion_{time.strftime('%Y%m%d_%H%M%S')}_s{seed}.png"
    path = os.path.join(out_dir, fname)
    try:
        img.save(path)
        return os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        st.warning(f"Could not save output: {e}")
        return None


@st.cache_resource(show_spinner=False)
def assets_dirs():
    from core import config_model_dirs
    return config_model_dirs()


def _controlnet_dirs(assets):
    """Return controlnet model directories in models/controlnet (diffusers dirs)."""
    cn_dir = assets["dirs"]["controlnet"]
    if not os.path.isdir(cn_dir):
        return []
    return [d for d in sorted(os.listdir(cn_dir))
            if os.path.isdir(os.path.join(cn_dir, d))]


def _make_control_map(img, method, low, high, levels):
    try:
        if method == "PyraCanny":
            return preprocess(img, "PyraCanny", low=low, high=high, levels=levels)
        return preprocess(img, "Canny", low=low, high=high)
    except Exception as e:
        st.warning(f"{method} failed: {e}")
        return None


@st.cache_resource(show_spinner=False)
def _cached_controlnet_pipeline(model_path, vae_path, controlnet_path, architecture):
    backend = detect_backend()
    vram_state = get_vram_state(backend, detect_vram_mb(backend))
    pipe = load_controlnet_pipeline(model_path, vae_path, controlnet_path, backend, architecture)
    return pipe, backend, vram_state


def _refresh_controlnet_pipeline(cfg, assets, prefer=None):
    dirs = _controlnet_dirs(assets)
    if not dirs:
        st.error("No ControlNet model directory found in models/controlnet/. "
                 "Run `python setup.py` to download one, or add it manually.")
        return None, None, None
    # Pick a controlnet dir: prefer one whose name contains `prefer` (e.g. 'canny'
    # or 'pose'), else fall back with a clear notice.
    chosen = None
    if prefer:
        for d in dirs:
            if prefer.lower() in d.lower():
                chosen = d
                break
    if chosen is None:
        if prefer:
            st.warning(f"No '{prefer}' ControlNet found in models/controlnet/. "
                       f"Run `python setup.py` to download it. Using '{dirs[0]}' instead, "
                       "which may not match this tab.")
        chosen = dirs[0]
    d = assets["dirs"]
    ckpt_path = os.path.join(d["checkpoints"], cfg.checkpoint)
    vae_path = os.path.join(d["vae"], cfg.vae) if cfg.vae else default_vae_path(d["vae"])
    controlnet_path = os.path.join(d["controlnet"], chosen)
    key = (cfg.checkpoint, cfg.vae, cfg.architecture, chosen)
    is_new = st.session_state.get("vd_cn_key") != key
    if is_new:
        _cached_controlnet_pipeline.clear()
        st.session_state.vd_cn_key = key
    try:
        if is_new:
            with st.spinner(f"Loading ControlNet pipeline ('{chosen}') into memory — "
                            "first load after a model change is slow..."):
                return _cached_controlnet_pipeline(ckpt_path, vae_path, controlnet_path, cfg.architecture)
        return _cached_controlnet_pipeline(ckpt_path, vae_path, controlnet_path, cfg.architecture)
    except Exception as e:
        st.error(f"Failed to load ControlNet pipeline: {e}")
        return None, None, None


def _detect_pose(img):
    from core import detect_pose_from_image
    from core.catalog import by_key
    entry = by_key("openpose_annotator")
    if entry is not None and not entry.is_present(os.path.dirname(os.path.abspath(__file__))):
        st.info("OpenPose annotator weights not found — downloading (~200 MB) on first use. "
                "This one-time step makes the first pose detection slow. "
                "Tip: run `python setup.py` to fetch it in advance.")
    try:
        return detect_pose_from_image(img)
    except Exception as e:
        st.warning(f"OpenPose failed: {e}")
        return None


def _skeleton_from_json(data, width, height):
    from core import skeleton_from_json
    try:
        return skeleton_from_json(data, width, height)
    except Exception as e:
        st.error(f"Invalid OpenPose JSON: {e}")
        return None


def _caption(img):
    """Caption an image via BLIP (results are memoized per-upload in session_state)."""
    from core import caption_image
    return caption_image(img)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.markdown(UI_CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="vd-title">Visual Diffusion Studio</h1>', unsafe_allow_html=True)

    if "app_config" not in st.session_state:
        st.session_state.app_config = AppConfig.from_config_file()
        # Ensure a valid checkpoint is set (config file may omit it).
        if not st.session_state.app_config.checkpoint:
            st.session_state.app_config.checkpoint = DEFAULT_CHECKPOINT
        st.session_state.app_config.loras = [("", 0.0), ("", 0.0)]

    assets = discover_all()
    if not assets["checkpoints"]:
        st.error("No checkpoint models found in models/checkpoints/.")
        return

    cfg = universal_sidebar(assets)
    st.session_state.app_config = cfg
    preset_panel()

    if st.sidebar.button("Quit App", type="secondary", help="Stop the Streamlit server"):
        os._exit(0)

    tabs = st.tabs([
        "Text → Image", "Text → Pixel Art", "Image → Text", "Image → Image", "Image Control",
        "IP Consistency", "Pose Control",
    ])
    with tabs[0]:
        tab_text_to_image(assets)
    with tabs[1]:
        tab_pixel_art(assets)
    with tabs[2]:
        tab_image_to_text(assets)
    with tabs[3]:
        tab_image_to_image(assets)
    with tabs[4]:
        tab_image_control(assets)
    with tabs[5]:
        tab_ip_consistency(assets)
    with tabs[6]:
        tab_pose_control(assets)


if __name__ == "__main__":
    main()
