# VisualDiffusion

A Streamlit-based image generation studio with a mobile-first 9:16 share workflow.

## Prerequisites

- Python 3.12
- pip
- A machine with a CUDA GPU, or CPU (Windows recommended with an NVIDIA GPU)

## Setup from Scratch

### 1. Clone the repository

   ```bash
   git clone https://github.com/MikeSourceCode/VisualDiffusionWindows.git
   cd VisualDiffusionWindows
   ```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3.12 -m venv visual
source visual/bin/activate
```

**Windows:**
```bash
python -m venv visual
visual\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** PyTorch is installed separately by `setup.py` with the correct CUDA variant. Do **not** install the CPU-only `torch` from `requirements.txt` — run `python setup.py` first.

### 4. Run first-time setup

This creates required directories, installs PyTorch (with CUDA if an NVIDIA GPU is detected), configures safety settings, and walks through generation defaults.

```bash
python setup.py
```

**GPU detection:** `setup.py` auto-detects NVIDIA GPUs via `nvidia-smi`. If a GPU is found, it offers to install the CUDA-accelerated PyTorch build; otherwise it installs the CPU-only build.

**Safety setup notes:**
- The NSFW image blackout checker defaults to **enabled**. To disable it, you must type `FALSE` when prompted.
- The text classifier prompt blocker defaults to **enabled**. To disable it, you must type `FALSE` when prompted. You must also provide a classifier model by setting `TEXT_CLASSIFIER_MODEL` in `config/app_config.json` or placing `model.safetensors` in `models/safety/`.
- Both settings can be reconfigured later by running `python setup.py --config` without re-installing anything.

### 5. Launch the app

```bash
streamlit run app.py
```

## Adding Models

No models are downloaded automatically. Place checkpoint files directly into the `models/` tree:

- **Single-file checkpoint** → `models/checkpoints/your_model.safetensors`
- **Full model set (folder-based)** → `models/model_set/<repo-name>/`

The app discovers both single-file checkpoints and full-model-set folders automatically.

### Downloading checkpoints manually (Windows curl)

```bat
curl -L -o models\checkpoints\model.safetensors "https://huggingface.co/<repo>/resolve/main/model.safetensors"
```

## Project Structure

- **`app.py`** — Main Streamlit application and generation tabs
- **`core/`** — Engine, config, safety, and prompt tag logic
- **`ui/`** — Share card and phone-frame rendering
- **`config/app_config.example.json`** — Operator configuration template
- **`config/app_config.json`** — Operator configuration (gitignored, created by `setup.py`)
- **`setup.py`** — First-time setup (GPU detection, safety prompts, config)
- **`getmodel.py`** — Deprecated; no longer downloads (place checkpoints manually)
- **`requirements.txt`** — Python dependencies

## Configuration

Operator-editable settings live in `config/app_config.json`. The template in `config/app_config.example.json` documents all available options, including:

- Checkpoint, VAE, LoRAs, steps, CFG scale, seed, output size
- Per-tab default prompts and negative prompts
- Safety settings: `safety_checker_black_out_nsfw`, `text_classifier_enabled`, `early_safety_check`, `early_safety_step`, `early_safety_step_frac`

## Safety Architecture

1. **Layer 1 — Blocked term list:** Operator-maintained list in `safety.py`
2. **Layer 2 — Text classifier:** Operator-configured model blocks flagged prompts before generation
3. **Layer 3 — Image censor:** CompVis StableDiffusionSafetyChecker replaces flagged output with a black image
4. **Early safety gate:** Single CLIP check at a configurable denoise step

## Known Todos

- Optimize pipeline load and LoRA switching latency between tabs
- Reduce click-to-first-preview latency on multi-GPU setups
- Improve mobile layout robustness across Streamlit versions

