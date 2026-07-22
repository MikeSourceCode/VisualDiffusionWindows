# VisualDiffusion

A Streamlit-based image generation studio with a mobile-first 9:16 share workflow.

No models, weights, or checkpoints are bundled. You provide the assets in `models/`; the
app runs entirely under your local control.

## Essential Setup

1. **Clone** the repo
2. **Create the venv** — `python3.12 -m venv visual && source visual/bin/activate`
3. **Install dependencies** — `pip install -r requirements.txt`
4. **Run setup** — `python setup.py` (creates directories, downloads curated models, configures safety)
5. **Launch** — `streamlit run app.py`

For full step-by-step setup, model placement, and operator configuration, see **[SETUP.md](SETUP.md)**.

## Requirements

- Python 3.12
- pip
- A machine with a CUDA, MPS (Apple Silicon), or CPU GPU backend

## Project Structure

- **`app.py`** — Main Streamlit application and generation tabs
- **`core/`** — Engine, config, safety, and asset discovery
- **`ui/`** — Share card and phone-frame rendering
- **`config/app_config.example.json`** — Configuration template
- **`config/app_config.json`** — Live operator config (gitignored, created by `setup.py`)
- **`setup.py`** — First-time setup, model downloads, and safety prompts
- **`requirements.txt`** — Python dependencies

## Configuration

`config/app_config.json` is the live config file. It is **not** committed to git; the
template `config/app_config.example.json` documents every available option and is versioned.

Key sections:

- **`common`** — Checkpoint, VAE, LoRAs, steps, CFG scale, seed, output size, previews
- **`tabs`** — Per-tab default prompts, negative prompts, and tab-specific overrides
- **Safety** — `safety_checker_black_out_nsfw`, `text_classifier_enabled`, `early_safety_*`

On first run, or any time you run `python setup.py --config`, the app will walk you
through every parameter to create or update the config from scratch.

## Safety & Operator Responsibility

All content moderation lives in `safety.py` and is enforced in four layers:

1. **Prompt term gate** — operator-maintained blocklist in `safety.py`; prompts in non-Latin scripts are refused
2. **Text classifier** — optional prompt-side NSFW check before generation starts
3. **Image censor** — CompVis StableDiffusionSafetyChecker replaces flagged output with a black image
4. **Early safety gate** — single CLIP check at a configurable denoise step to abort unsafe content early

Safety settings require an intentional operator decision to opt out. They are configured
via `python setup.py --config` and are not exposed in the Streamlit sidebar.

See `safety.py` and `models/safety_checker/README.md` for details on each layer.

## Model Placement

- Checkpoints → `models/checkpoints/`
- VAEs → `models/vae/`
- LoRAs → `models/lora/`
- ControlNets → `models/controlnet/`
- IP-Adapters → `models/ip-adapter/`
- Safety checkers → `models/safety_checker/`

Missing assets are detected at runtime; the app will prompt you to download them on first use.
