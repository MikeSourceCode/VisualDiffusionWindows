# `models/unet/`

UNet config and weights used by the pipeline when loading checkpoints in diffusers format.
The engine does not require this folder for single-file `.safetensors` checkpoint loading,
but it is used when a checkpoint is split into separate components.

> None of these files are committed. Download the ones you want below.

## Files used during development

| Filename | Purpose | Source |
|---|---|---|
| `unet/config.json` | UNet architecture config (SDXL) | HF `stabilityai/stable-diffusion-xl-base-1.0` |
| `unet/diffusion_pytorch_model.safetensors` | UNet weights (SDXL) | HF `stabilityai/stable-diffusion-xl-base-1.0` |

When loading a checkpoint via `DiffusersPipeline.from_pretrained()`, the UNet files are
fetched automatically from Hugging Face. Place them here only if you want offline loading or
to override the checkpoint's baked-in UNet.

## Download

```bash
# SDXL UNet (Hugging Face)
curl -L -o unet/diffusion_pytorch_model.safetensors \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors?download=true"

curl -L -o unet/config.json \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/config.json?download=true"
```
