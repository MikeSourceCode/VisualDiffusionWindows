# `models/lora/`

LoRA adapters. The engine loads these **UNet-only** (A1111-format LoRAs crash the SDXL
text-encoder path in this diffusers version), with a per-LoRA weight applied at load time.
Select and weight them via `TEST_LORAS` in `latent_test_devices.py`, e.g.:

```python
TEST_LORAS = [("add-detail-xl.safetensors", 0.6), ("touch-of-realism-sdxl.safetensors", 0.5)]
```

> None of these files are committed. Download the ones you want below.

## LoRAs used during development

| Filename | Purpose | Source |
|---|---|---|
| `add-detail-xl.safetensors` | Detail tweaker (Detail Tweaker XL) | HF `LyliaEngine/add-detail-xl` / Civitai `122359` |
| `xl_more_art-full-xl_real-enhancer.safetensors` | Art / realism enhancer | HF `frankjoshua/xl_more_art-full_v1` / Civitai `124347` |
| `touch-of-realism-sdxl.safetensors` | Photographic realism (subtle) | Civitai `1705430` |
| `pony-add-more-details.safetensors` | Detail booster for Pony | Civitai `669571` |
| `realism_lora_yogiv3.safetensors` | Realism style | Unverified source — search Civitai/HF for "realism lora yogi v3" |

## Download

```bash
# Detail Tweaker XL  (Hugging Face)
curl -L -o add-detail-xl.safetensors \
  "https://huggingface.co/LyliaEngine/add-detail-xl/resolve/main/add-detail-xl.safetensors?download=true"

# xl_more_art-full v1  (Hugging Face; rename to the project's filename)
curl -L -o xl_more_art-full-xl_real-enhancer.safetensors \
  "https://huggingface.co/frankjoshua/xl_more_art-full_v1/resolve/main/xl_more_art-full_v1.safetensors?download=true"

# --- Civitai (add ?token=$CIVITAI_TOKEN if required) ---

# Touch of Realism [SDXL] v2
curl -L -o touch-of-realism-sdxl.safetensors \
  "https://civitai.com/api/download/models/1934796"

# Pony Add more details v1.0
curl -L -o pony-add-more-details.safetensors \
  "https://civitai.com/api/download/models/749546"
```

> `realism_lora_yogiv3.safetensors` could not be confidently traced to a public source.
> If you have it, drop it here; otherwise substitute any SDXL realism LoRA and update `TEST_LORAS`.
