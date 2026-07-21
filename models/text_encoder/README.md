# `models/text_encoder/`

CLIP text encoder weights used by the pipeline when loading checkpoints in diffusers format.
The engine does not require this folder for single-file `.safetensors` checkpoint loading,
but it is used when a checkpoint is split into separate components.

> None of these files are committed. Download the ones you want below.

## Files used during development

| Filename | Purpose | Source |
|---|---|---|
| `text_encoder.safetensors` | CLIP ViT-L/14 text encoder (SDXL) | HF `stabilityai/stable-diffusion-xl-base-1.0` |
| `text_encoder_2.safetensors` | OpenCLIP ViT-bigG/14 text encoder (SDXL) | HF `stabilityai/stable-diffusion-xl-base-1.0` |

When loading a checkpoint via `DiffusersPipeline.from_pretrained()`, the text encoder files are
fetched automatically from Hugging Face. Place them here only if you want offline loading or
to override the checkpoint's baked-in text encoders.

## Download

```bash
# SDXL text encoders (Hugging Face)
curl -L -o text_encoder/text_encoder.safetensors \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.safetensors?download=true"

curl -L -o text_encoder/text_encoder_2.safetensors \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors?download=true"
```
