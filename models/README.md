# `models/`

Model weights for VisualDiffusion. **Nothing in this folder is committed to git** except
these `README.md` files — the `.gitignore` excludes all `.safetensors`, `.ckpt`, `.bin`,
`config.json`, and tokenizer files. Each subfolder's README lists the files the project was
developed against and how to download them, so anyone who forks the repo can choose which
models to pull.

## Folder layout

| Folder | Purpose |
|---|---|
| `checkpoints/` | Full SDXL / SD 1.5 base models (single-file `.safetensors`). |
| `lora/` | LoRA adapters, loaded UNet-only at generation time. |
| `vae/` | Standalone VAE weights (used to override the checkpoint's baked VAE). |
| `ip_adapter/` | IP-Adapter weights for identity-preserving image conditioning. |
| `tokenizer/` | Diffusers-format tokenizer files (optional; for diffusers-layout loading). |
| `text_encoder/` | Diffusers-format CLIP text encoder weights (optional). |
| `unet/` | Diffusers-format UNet config/weights (optional). |

The engine (`latent_test_devices.py`) loads from `checkpoints/`, `vae/`, and `lora/` via
`from_single_file`. The `tokenizer/`, `text_encoder/`, and `unet/` folders hold optional
diffusers-format components and are documented for completeness.

## Downloading

Two common sources are used below:

- **Hugging Face** — public files download directly with `curl`:
  ```bash
  curl -L -o <output-file> "https://huggingface.co/<repo>/resolve/main/<path>?download=true"
  ```
  Gated repos need a token: add `-H "Authorization: Bearer $HF_TOKEN"`.

- **Civitai** — download by model *version* id:
  ```bash
  curl -L -o <output-file> "https://civitai.com/api/download/models/<versionId>"
  ```
  Many Civitai models require an account/API token:
  add `?token=$CIVITAI_TOKEN` to the URL. Create one at civitai.com → Account settings → API Keys.

Run the `curl` commands **from inside the matching subfolder**, or pass a full path to `-o`.

## Notes

- File **names matter**: the engine references models by the exact filenames listed in each
  subfolder's README. If you download a differently named file, rename it to match or update
  the `TEST_*` constants in `latent_test_devices.py`.
- Check each model's **license** on its source page before redistributing or using commercially.
