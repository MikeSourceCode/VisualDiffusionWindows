#!/usr/bin/env python3
"""
Setup script for the Streamlit Image Generation App
-------------------------------------------------
- Creates required folders
- Optionally installs requirements.txt
- Downloads official SDXL Base 1.0 (safest starting point)
- Downloads official SDXL VAE
- Lets the operator add extra models / VAEs / LoRAs
- Puts responsibility for ethical use on the operator
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.catalog import CATALOG, CatalogEntry

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def ensure_dirs():
    for d in ("models/checkpoints", "models/vae", "models/lora",
              "models/controlnet", "models/annotators",
              "models/safety_checker", "models/safety", "outputs"):
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Folders ready: models/{checkpoints,vae,lora,controlnet,annotators,safety_checker,safety}, outputs\n")


def install_requirements():
    print("=" * 60)
    print("STEP 0 – Install Python dependencies")
    print("=" * 60)
    try:
        import streamlit, diffusers, transformers, torch, PIL
        print("✓ Core dependencies already installed (skipping requirements install).\n")
        return
    except ImportError:
        pass
    choice = input("Install requirements.txt now? [Y/n]: ").strip().lower()
    if choice in ("", "y", "yes"):
        print("Installing requirements... (this may take a minute)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed\n")
    else:
        print("Skipped requirements installation.\n")


def download_catalog_entry(entry: CatalogEntry) -> bool:
    """Download one catalog entry (specific files, whole-repo, or annotator)."""
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"\nDownloading {entry.desc} ({entry.size})...")
    try:
        if entry.kind == "annotator":
            # controlnet_aux caches the annotator in the HF cache on first load.
            from controlnet_aux import OpenposeDetector
            OpenposeDetector.from_pretrained(entry.repo)
            print("✓ Annotator cached (Hugging Face cache)")
            return True

        dest = entry.dest_abs(root)
        Path(dest).mkdir(parents=True, exist_ok=True)
        if entry.files:
            for fn in entry.files:
                p = hf_hub_download(repo_id=entry.repo, filename=fn, local_dir=dest,
                                    local_dir_use_symlinks=False)
                mb = os.path.getsize(p) // (1024 * 1024)
                print(f"  ✓ {fn} ({mb} MB)")
        else:
            snapshot_download(repo_id=entry.repo, local_dir=dest, local_dir_use_symlinks=False)
            print(f"  ✓ snapshot → {dest}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def prompt_catalog():
    """Iterate the model catalog; offer to download each missing asset."""
    root = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("STEP 1 – App models (download now or skip; you can rerun setup later)")
    print("=" * 60)
    for entry in CATALOG:
        present = entry.is_present(root)
        status = "already present" if present else "NOT downloaded"
        print(f"\n• {entry.desc}")
        print(f"    for: {entry.required_for} | size: {entry.size} | status: {status}")
        if present:
            print("    → skipping (already on disk)")
            continue
        choice = input(f"    Download now? [Y/n] (ENTER=yes): ").strip().lower()
        if choice in ("", "y", "yes"):
            download_catalog_entry(entry)
        else:
            print("    → skipped. The app will download/first-load it on demand "
                  "(the first such generation will be slower).")


def _prompt_common(models_root: str, existing: dict) -> dict:
    """Interactively build the `common` section of app_config.json."""
    from core.assets import scan_checkpoints, scan_subfolder
    common = dict(existing.get("common", {}))

    def _choose(label, options, default, fallback=""):
        opts = [fallback] + options
        default_display = default if default in opts else fallback
        try:
            idx = opts.index(default_display)
        except ValueError:
            idx = 0
        for i, o in enumerate(opts):
            print(f"  {i + 1}. {o or '(none)'}")
        choice = input(f"{label} [default={default_display or '(none)'}]: ").strip()
        if not choice:
            return default_display
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(opts):
                return opts[idx]
        return choice

    print("\n--- Checkpoint ---")
    ckpts = list(scan_checkpoints(os.path.join(models_root, "checkpoints")).keys())
    common["checkpoint"] = _choose("Select checkpoint", ckpts, common.get("checkpoint", ""))

    if common["checkpoint"]:
        from core.assets import detect_architecture
        arch = detect_architecture(os.path.join(models_root, "checkpoints", common["checkpoint"]))
        common.setdefault("architecture", arch)
        print(f"  → Detected architecture: {arch}")

    print("\n--- VAE ---")
    vaes = scan_subfolder(os.path.join(models_root, "vae"))
    common["vae"] = _choose("Select VAE (blank = built-in)", vaes, common.get("vae", ""), fallback="")

    print("\n--- LoRAs ---")
    loras = scan_subfolder(os.path.join(models_root, "lora"))
    current = common.get("loras", [])
    if not isinstance(current, list):
        current = []
    while len(current) < 2:
        current.append(["", 0.0])
    for slot in range(2):
        name = current[slot][0] if slot < len(current) else ""
        weight = current[slot][1] if slot < len(current) else 0.0
        lname = _choose(f"LoRA {slot + 1} name", loras, name, fallback="")
        if lname:
            lw = input(f"  LoRA {slot + 1} weight [{weight}]: ").strip()
            try:
                weight = float(lw) if lw else weight
            except ValueError:
                pass
            current[slot] = [lname, weight]
        else:
            current[slot] = ["", 0.0]
    common["loras"] = current[:2]

    print("\n--- Generation ---")
    common["steps"] = int(input(f"Steps [{common.get('steps', 20)}]: ") or common.get("steps", 20))
    common["cfg_scale"] = float(input(f"CFG Scale [{common.get('cfg_scale', 7.0)}]: ") or common.get("cfg_scale", 7.0))
    common["seed"] = int(input(f"Seed (0=random) [{common.get('seed', 0)}]: ") or common.get("seed", 0))
    common["sharpness"] = float(input(f"Sharpness [{common.get('sharpness', 0.0)}]: ") or common.get("sharpness", 0.0))
    common["strength"] = float(input(f"Img2Img Strength [{common.get('strength', 0.65)}]: ") or common.get("strength", 0.65))

    print("\n--- Output Size (Text→Image) ---")
    print("  1. 9:16 Portrait (768x1368)")
    print("  2. 1:1 Square (1024x1024)")
    print("  3. 16:9 Landscape (1472x832)")
    cur_w = common.get("width", 768)
    cur_h = common.get("height", 1368)
    choice = input(f"Output size [default={cur_w}x{cur_h}]: ").strip()
    size_map = {"1": (768, 1368), "2": (1024, 1024), "3": (1472, 832)}
    if choice in size_map:
        common["width"], common["height"] = size_map[choice]
    elif choice.isdigit() and int(choice) in (1, 2, 3):
        common["width"], common["height"] = size_map[choice]
    else:
        common.setdefault("width", cur_w)
        common.setdefault("height", cur_h)

    print("\n--- Preview ---")
    show = common.get("show_previews", True)
    ans = input(f"Show live previews? [Y/n] (ENTER={'yes' if show else 'no'}): ").strip().lower()
    if ans in ("n", "no"):
        common["show_previews"] = False
    else:
        common["show_previews"] = True
    common["preview_frequency"] = int(
        input(f"Preview every N steps [{common.get('preview_frequency', 1)}]: ") or common.get("preview_frequency", 1)
    )

    print("\n--- ControlNet / IP-Adapter ---")
    common["controlnet_strength"] = float(
        input(f"ControlNet strength [{common.get('controlnet_strength', 1.0)}]: ") or common.get("controlnet_strength", 1.0)
    )
    common["ip_adapter_scale"] = float(
        input(f"IP-Adapter scale [{common.get('ip_adapter_scale', 0.6)}]: ") or common.get("ip_adapter_scale", 0.6)
    )

    print("\n--- Prompt Defaults ---")
    common["default_prompt"] = input(f"Default positive prompt [{common.get('default_prompt', '')}]: ") or common.get("default_prompt", "")
    common["default_negative_prompt"] = input(f"Default negative prompt [{common.get('default_negative_prompt', '')}]: ") or common.get("default_negative_prompt", "")

    return common


def _prompt_tabs(existing: dict) -> dict:
    """Interactively build per-tab prompt defaults."""
    tabs = {}
    tab_names = {
        "text_to_image": "Text → Image",
        "image_to_image": "Image → Image",
        "image_control": "Image Control",
        "ip_consistency": "IP Consistency",
        "image_to_text": "Image → Text",
        "pose_control": "Pose Control",
    }
    for key, label in tab_names.items():
        existing_tab = (existing.get("tabs") or {}).get(key, {})
        if existing_tab:
            print(f"\n--- {label} ---")
            dp = input(f"  Positive prompt [{existing_tab.get('default_prompt', '')}]: ") or existing_tab.get("default_prompt", "")
            dn = input(f"  Negative prompt [{existing_tab.get('default_negative_prompt', '')}]: ") or existing_tab.get("default_negative_prompt", "")
            tabs[key] = {"default_prompt": dp, "default_negative_prompt": dn}
            if key == "image_to_image":
                tabs[key]["strength"] = float(input(f"  Strength [{existing_tab.get('strength', 0.65)}]: ") or existing_tab.get("strength", 0.65))
            elif key == "image_control":
                tabs[key]["controlnet_strength"] = float(input(f"  ControlNet strength [{existing_tab.get('controlnet_strength', 1.0)}]: ") or existing_tab.get("controlnet_strength", 1.0))
            elif key == "ip_consistency":
                tabs[key]["ip_adapter_scale"] = float(input(f"  IP-Adapter scale [{existing_tab.get('ip_adapter_scale', 0.6)}]: ") or existing_tab.get("ip_adapter_scale", 0.6))
        else:
            tabs[key] = existing_tab
    return tabs


def write_operator_config():
    """Create/update config/app_config.json with operator choices."""
    from core.config import config_file_path, save_config_file, load_config_file

    existing = load_config_file()
    print("\n" + "=" * 60)
    print("CONFIG UPDATE – Adjust settings")
    print("=" * 60)

    models_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    # --- Safety blackout gate ---
    current_raw = existing.get("safety_checker_black_out_nsfw", True)
    if isinstance(current_raw, str):
        current = current_raw.strip().upper() == "FALSE"
    else:
        current = bool(current_raw)
    print(
        "\nThe app includes a safety checker that replaces flagged NSFW images\n"
        "with a black image (the same checker used by Stable Diffusion).\n"
    )
    if current:
        choice = input("Keep blackout censoring ENABLED? [Y/n] (ENTER=yes): ").strip().lower()
        if choice in ("", "y", "yes"):
            existing["safety_checker_black_out_nsfw"] = True
            print("→ Safety blackout stays ENABLED.\n")
        else:
            print("You chose to DISABLE the safety blackout.")
            confirmed = False
            while not confirmed:
                answer = input("> Type FALSE to confirm disabling the safety checker: ").strip()
                if answer == "FALSE":
                    confirmed = True
                    existing["safety_checker_black_out_nsfw"] = False
                    print("→ Safety blackout DISABLED. You typed FALSE; this is recorded.\n")
                else:
                    print("  That is not the literal FALSE. Safety blackout stays ENABLED.")
                    existing["safety_checker_black_out_nsfw"] = True
                    break
    else:
        choice = input("Safety blackout is currently DISABLED. Re-enable it? [Y/n] (ENTER=yes): ").strip().lower()
        if choice in ("", "y", "yes"):
            existing["safety_checker_black_out_nsfw"] = True
            print("→ Safety blackout RE-ENABLED.\n")
        else:
            print("→ Safety blackout stays DISABLED.\n")

    # --- Text classifier gate ---
    current_raw = existing.get("text_classifier_enabled", True)
    if isinstance(current_raw, str):
        current = current_raw.strip().upper() == "TRUE"
    else:
        current = bool(current_raw)
    print(
        "\nThe app includes a text classifier that blocks prompts flagged as NSFW\n"
        "before generation starts. The operator must provide a classifier model\n"
        "and set TEXT_CLASSIFIER_MODEL accordingly.\n"
    )
    if current:
        choice = input("Keep text-classifier prompt blocking ENABLED? [Y/n] (ENTER=yes): ").strip().lower()
        if choice in ("", "y", "yes"):
            existing["text_classifier_enabled"] = True
            print("→ Text-classifier prompt blocking stays ENABLED.\n")
        else:
            print("You chose to DISABLE the text-classifier prompt blocking.")
            confirmed = False
            while not confirmed:
                answer = input("> Type FALSE to confirm disabling: ").strip()
                if answer == "FALSE":
                    confirmed = True
                    existing["text_classifier_enabled"] = False
                    print("→ Text-classifier prompt blocking DISABLED. You typed FALSE; this is recorded.\n")
                else:
                    print("  That is not the literal FALSE. Text-classifier prompt blocking stays ENABLED.")
                    existing["text_classifier_enabled"] = True
                    break
    else:
        choice = input("Text-classifier prompt blocking is currently DISABLED. Re-enable it? [Y/n] (ENTER=yes): ").strip().lower()
        if choice in ("", "y", "yes"):
            existing["text_classifier_enabled"] = True
            print("→ Text-classifier prompt blocking RE-ENABLED.\n")
        else:
            print("→ Text-classifier prompt blocking stays DISABLED.\n")

    # --- Interactive generation defaults ---
    if "common" not in existing and "tabs" not in existing:
        print("\nNo configuration found. Let's create one from scratch.")
        common = _prompt_common(models_root, {})
        existing["common"] = common
        tabs = _prompt_tabs({})
        existing["tabs"] = tabs
    else:
        ans = input("\nUpdate generation defaults? [Y/n] (ENTER=yes): ").strip().lower()
        if ans not in ("n", "no"):
            common = _prompt_common(models_root, existing)
            existing["common"] = common
            tabs = _prompt_tabs(existing)
            existing["tabs"] = tabs
        else:
            print("→ Skipping generation defaults update.\n")

    # Back-compat migration for old flat keys...
    if "common" not in existing:
        existing["common"] = existing.get("common", {})

    # Merge missing defaults from the example config so common + tabs are fully populated.
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "app_config.example.json")
    try:
        with open(example_path, "r", encoding="utf-8") as f:
            example_defaults = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        example_defaults = {}

    def _merge_missing(base, defaults):
        for k, v in defaults.items():
            if k not in base:
                base[k] = v
            elif isinstance(base[k], dict) and isinstance(v, dict):
                _merge_missing(base[k], v)
        return base

    _merge_missing(existing, example_defaults)

    save_config_file(existing)
    print(f"✓ Config written → {config_file_path()}\n")
    if "common" not in existing and "tabs" not in existing:
        common_keys = {
            "checkpoint", "vae", "loras", "steps", "cfg_scale", "seed",
            "sharpness", "strength", "width", "height", "show_previews",
            "preview_frequency", "controlnet_strength", "ip_adapter_scale",
            "default_prompt", "default_negative_prompt",
        }
        common = {k: existing[k] for k in common_keys if k in existing}
        tabs = {}
        tab_map = {
            "text_to_image": "t2i_default_prompt",
            "image_to_image": "i2i_default_prompt",
            "image_control": "ic_default_prompt",
            "ip_consistency": "ip_default_prompt",
            "image_to_text": "i2t_default_prompt",
            "pose_control": "pose_default_prompt",
        }
        neg_map = {
            "text_to_image": "t2i_default_negative",
            "image_to_image": "i2i_default_negative",
            "image_control": "ic_default_negative",
            "ip_consistency": "ip_default_negative",
            "image_to_text": "i2t_default_negative",
            "pose_control": "pose_default_negative",
        }
        for tab, pk in tab_map.items():
            if pk in existing or neg_map.get(tab) in existing:
                tabs[tab] = {}
                if pk in existing:
                    tabs[tab]["default_prompt"] = existing[pk]
                if neg_map.get(tab) in existing:
                    tabs[tab]["default_negative_prompt"] = existing[neg_map[tab]]
        if common:
            existing["common"] = common
        if tabs:
            existing["tabs"] = tabs

    # Migrate any lingering flat prompt keys into tabs if the tab entry is empty.
    flat_prompt_map = {
        "text_to_image": "t2i_default_prompt",
        "image_to_image": "i2i_default_prompt",
        "image_control": "ic_default_prompt",
        "ip_consistency": "ip_default_prompt",
        "image_to_text": "i2t_default_prompt",
        "pose_control": "pose_default_prompt",
    }
    flat_negative_map = {
        "text_to_image": "t2i_default_negative",
        "image_to_image": "i2i_default_negative",
        "image_control": "ic_default_negative",
        "ip_consistency": "ip_default_negative",
        "image_to_text": "i2t_default_negative",
        "pose_control": "pose_default_negative",
    }
    tabs_container = existing.setdefault("tabs", {})
    for tab, pk in flat_prompt_map.items():
        if pk in existing:
            tabs_container.setdefault(tab, {})
            if not tabs_container[tab].get("default_prompt"):
                tabs_container[tab]["default_prompt"] = existing[pk]
    for tab, nk in flat_negative_map.items():
        if nk in existing:
            tabs_container.setdefault(tab, {})
            if not tabs_container[tab].get("default_negative_prompt"):
                tabs_container[tab]["default_negative_prompt"] = existing[nk]

    # Merge missing defaults from the example config so common + tabs are fully populated.
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "app_config.example.json")
    try:
        with open(example_path, "r", encoding="utf-8") as f:
            example_defaults = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        example_defaults = {}

    def _merge_missing(base, defaults):
        for k, v in defaults.items():
            if k not in base:
                base[k] = v
            elif isinstance(base[k], dict) and isinstance(v, dict):
                _merge_missing(base[k], v)
        return base

    _merge_missing(existing, example_defaults)

    save_config_file(existing)
    print(f"✓ Config written → {config_file_path()}\n")


def download_file(repo_id: str, filename: str, local_dir: str, description: str):
    """Download a single file and return the local path."""
    print(f"\nDownloading {description}...")
    print(f"  repo_id  : {repo_id}")
    print(f"  filename : {filename}")
    print(f"  target   : {local_dir}/")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Downloaded → {path}")
        return path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None


def prompt_extra_download(local_dir: str, kind: str = "model"):
    """
    Let the user enter additional downloads while the main one is happening
    or after it finishes. Empty line = skip.
    """
    print("\n" + "-" * 50)
    print(f"You can download extra {kind}s now.")
    print("Format example:")
    print("  repo_id/filename")
    print("  e.g.  stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors")
    print("  or just press ENTER to skip and continue.")
    print("-" * 50)

    while True:
        user_input = input(f"Extra {kind} (repo_id/filename) or ENTER to skip: ").strip()
        if not user_input:
            print("→ Skipping extra downloads.\n")
            break

        try:
            if "/" not in user_input:
                print("Invalid format. Please use: repo_id/filename")
                continue

            # Split only on the last slash so repo_id can contain /
            *repo_parts, filename = user_input.rsplit("/", 1)
            repo_id = "/".join(repo_parts)

            download_file(repo_id, filename, local_dir, f"extra {kind}")
        except Exception as e:
            print(f"Error: {e}")


# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  STREAMLIT IMAGE APP – SAFE SETUP")
    print("=" * 60)
    print(
        "\nIMPORTANT – OPERATOR RESPONSIBILITY\n"
        "You are solely responsible for how this software and any models\n"
        "you download are used. Always follow applicable laws, platform\n"
        "policies, and ethical guidelines. Do not generate illegal,\n"
        "harmful, or non-consensual content.\n"
    )
    input("Press ENTER to continue if you accept this responsibility...")

    ensure_dirs()

    # ---------- 0.5 Operator config (safety + defaults) ----------
    write_operator_config()

    # ---------- 0. Requirements ----------
    install_requirements()

    # ---------- 1. Curated app models (from core/catalog.py) ----------
    prompt_catalog()

    # ---------- 2. Extra user-supplied assets ----------
    print("\n" + "=" * 60)
    print("STEP 2 – Add your own extra models / VAEs / LoRAs (optional)")
    print("=" * 60)
    prompt_extra_download("models/checkpoints", kind="checkpoint")
    prompt_extra_download("models/vae", kind="VAE")
    prompt_extra_download("models/lora", kind="LoRA")

    # ---------- Done ----------
    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETE")
    print("=" * 60)
    print(
        "You can now start the Streamlit app with:\n"
        "    streamlit run app.py\n\n"
        "Skipped a model? The app will download or first-load it on demand and\n"
        "tell you when it does (that first generation will be slower). You can\n"
        "also rerun `python setup.py` any time to fetch it up front.\n\n"
        "Remember: ethical use is the operator's responsibility.\n"
    )


def run_config_only():
    """Update only the operator config (safety + defaults), skip models and requirements."""
    from core.config import config_file_path, save_config_file, load_config_file
    existing = load_config_file()
    print("\n" + "=" * 60)
    print("CONFIG UPDATE – Adjust safety and defaults only")
    print("=" * 60)
    write_operator_config()
    print(f"✓ Config updated → {config_file_path()}\n")


def list_catalog():
    """Print catalog status without prompting (python setup.py --list)."""
    root = os.path.dirname(os.path.abspath(__file__))
    print("\nModel catalog status:\n")
    print(f"  {'KEY':<28}{'PRESENT':<9}{'SIZE':<10}DESCRIPTION")
    for e in CATALOG:
        print(f"  {e.key:<28}{('yes' if e.is_present(root) else 'NO'):<9}{e.size:<10}{e.desc}")
    print()


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_catalog()
    elif "--config" in sys.argv:
        run_config_only()
    else:
        main()