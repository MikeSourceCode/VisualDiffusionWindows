# `models/checkpoints/`

Full base models (SDXL / SD 1.5) as single-file `.safetensors`. The engine auto-detects the
architecture from the file header and picks the right pipeline. Set which one to use via
`TEST_CHECKPOINT` in `latent_test_devices.py`.

> None of these files are committed. Download the ones you want below.

## Models used during development

| Filename | Type | Source | Notes |
|---|---|---|---|
| `animaPencilXL_v500.safetensors` | SDXL (anime) | HF `bluepen5805/anima_pencil-XL` | Clean anime/illustration base. |
| `juggernautXL_v8Rundiffusion.safetensors` | SDXL (photoreal) | HF `RunDiffusion/Juggernaut-XL-v8` | Photorealistic. |
| `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | SDXL (photoreal) | HF `RunDiffusion/Juggernaut-XL-v9` | Photorealistic (v9). Works with the SDXL ControlNet + LoRA pipeline. |
| `ponyDiffusionV6XL.safetensors` | SDXL (Pony) | HF `Mistermango24/Pony-diffusion-xl-V6` / Civitai `257749` | Renamed from `ponyDiffusionV6XL_v6StartWithThisOne.safetensors`. Uses `score_*` tags. |
| `realisticStockPhoto_v20.safetensors` | SDXL (photoreal) | HF `AI2lab/SDXL-Models` / Civitai `139565` | Stock-photo realism. |

## Download

```bash
# anima_pencil-XL v5.0.0  (Hugging Face)
curl -L -o animaPencilXL_v500.safetensors \
  "https://huggingface.co/bluepen5805/anima_pencil-XL/resolve/main/anima_pencil-XL-v5.0.0.safetensors?download=true"

# Juggernaut XL v8  (Hugging Face)
curl -L -o juggernautXL_v8Rundiffusion.safetensors \
  "https://huggingface.co/RunDiffusion/Juggernaut-XL-v8/resolve/main/juggernautXL_v8Rundiffusion.safetensors?download=true"

# Juggernaut XL v9  (Hugging Face, ungated)
curl -L -o Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors \
  "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors?download=true"

# Pony Diffusion V6 XL  (Hugging Face mirror; note the rename to the short name)
curl -L -o ponyDiffusionV6XL.safetensors \
  "https://huggingface.co/Mistermango24/Pony-diffusion-xl-V6/resolve/main/ponyDiffusionV6XL_v6StartWithThisOne.safetensors?download=true"

# Realistic Stock Photo v2.0  (Hugging Face mirror)
curl -L -o realisticStockPhoto_v20.safetensors \
  "https://huggingface.co/AI2lab/SDXL-Models/resolve/main/realisticStockPhoto_v20.safetensors?download=true"
```

## Not added (and why)

- **DreamShaper XL** (`Lykon/DreamShaper_XL`): the canonical Lykon repos now return
  `401 Unauthorized` / appear gated or removed on Hugging Face, so they are not reliably
  redistributable from a verified first-party source. Skipped to keep sourcing reputable.
- **Stable Diffusion 3.5 Large** (`stabilityai/stable-diffusion-3.5-large`): (a) it is a
  **DiT / MM-DiT** architecture, not a U-Net SDXL checkpoint, so it will not load in the
  SDXL `from_single_file` + ControlNet + LoRA pipeline used here; and (b) it is **gated**
  (`gated: auto`) and requires an `HF_TOKEN` plus acceptance of the Stability Community
  license. Skipped. If you want it, it needs a separate MM-DiT pipeline and a licensed token.

Civitai alternatives (by version id): Pony `civitai.com/api/download/models/290740`,
Realistic Stock Photo `.../294470`.
