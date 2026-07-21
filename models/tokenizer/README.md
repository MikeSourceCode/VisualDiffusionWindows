# `models/tokenizer/`

Tokenizer files used by the text encoders when loading checkpoints in diffusers format.
The engine does not require this folder for single-file `.safetensors` checkpoint loading,
but it is used when a checkpoint is split into separate components.

> None of these files are committed. Download the ones you want below.

## Files used during development

| Filename | Purpose | Source |
|---|---|---|
| `tokenizer.json` | SDXL-compatible tokenizer (CLIP-G + CLIP-L merged) | HF `stabilityai/stable-diffusion-xl-base-1.0` |

When loading a checkpoint via `DiffusersPipeline.from_pretrained()`, the tokenizer files are
fetched automatically from Hugging Face. Place them here only if you want offline loading or
to override the checkpoint's baked-in tokenizer.

## Download

```bash
# SDXL tokenizer (Hugging Face)
curl -L -o tokenizer/tokenizer.json \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer.json?download=true"
```
