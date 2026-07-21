# `models/vae/`

Standalone VAE weights, used to override a checkpoint's baked VAE for cleaner colors. A
custom (user-supplied) VAE is loaded in **fp32** to avoid NaN/black-output issues on MPS; the
built-in checkpoint VAE stays **fp16** on MPS for fast decode (see `custom_vae_dtype` /
`get_vae_dtype` in `core/engine.py`). Point to one with `TEST_VAE` in `latent_test_devices.py`
(or leave `None` to use the checkpoint's built-in VAE).

> None of these files are committed. Download the ones you want below.

## Files used during development

| Filename | Purpose | Source |
|---|---|---|
| `diffusion_pytorch_model.safetensors` | SDXL VAE (fp16-safe), diffusers format | HF `madebyollin/sdxl-vae-fp16-fix` |
| `diffusion_pytorch_model.fp16.safetensors` | Same VAE, fp16 variant | HF `madebyollin/sdxl-vae-fp16-fix` |
| `config.json` | Diffusers VAE config (required beside the above) | HF `madebyollin/sdxl-vae-fp16-fix` |
| `ponyDiffusionV6XL_vae.safetensors` | SDXL VAE for Pony (single-file) | HF `Mistermango24/Pony-diffusion-xl-V6` (`sdxl_vae.safetensors`) |

`madebyollin/sdxl-vae-fp16-fix` is the standard SDXL VAE modified to run in fp16 without
producing NaNs — a good default for CUDA/ROCm. The Pony VAE is the plain SDXL VAE renamed.

## Download

```bash
# SDXL VAE (fp16-fix), diffusers format — needs config.json + weights together
curl -L -o config.json \
  "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json?download=true"
curl -L -o diffusion_pytorch_model.safetensors \
  "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors?download=true"

# Standalone SDXL VAE for Pony (single file; rename to the project's filename)
curl -L -o ponyDiffusionV6XL_vae.safetensors \
  "https://huggingface.co/Mistermango24/Pony-diffusion-xl-V6/resolve/main/sdxl_vae.safetensors?download=true"
```

> The `diffusion_pytorch_model.fp16.safetensors` filename is a diffusers `variant="fp16"` copy;
> the non-variant `diffusion_pytorch_model.safetensors` above works for this engine.
