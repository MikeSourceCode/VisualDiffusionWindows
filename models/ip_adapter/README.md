# `models/ip_adapter/`

IP-Adapter weights for identity-preserving image-to-image generation. The engine loads these
via `pipe.load_ip_adapter()` before generation. The IP-Adapter injects visual features from
a reference image into the cross-attention layers, allowing the model to lock a person's
identity while the prompt freely changes clothing, background, lighting, etc.

> None of these files are committed. Download the ones you want below.

## Files used during development

| Filename | Purpose | Source |
|---|---|---|
| `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` | IP-Adapter Plus for SDXL (face-aware) | HF `h94/IP-Adapter` → `sdxl_models/` |

`ip-adapter-plus_sdxl_vit-h` is the **face-aware** variant specifically designed to preserve
facial identity across style transfers and scene changes. It uses the CLIP ViT-H image encoder,
which `diffusers` fetches automatically when `image_encoder=None` is passed to `load_ip_adapter`.

## How it is used

Set these in `img_ip_adapter_test.py`:

```python
TEST_IP_ADAPTER = "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
TEST_IP_ADAPTER_SCALE = 0.6
```

The reference image (`TEST_INIT_IMAGE`, usually `~/Downloads/test.jpg`) is passed as both
the img2img starting point and the IP-Adapter identity reference. The model keeps the face
from that reference while your prompt (`TEST_PROMPT_VARIATION`) changes everything else.

## Download

```bash
# IP-Adapter Plus Face for SDXL ViT-H  (Hugging Face)
curl -L -o sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors?download=true"
```

> The `.safetensors` variant is preferred over the `.bin` pickle variant. Diffusers handles
> downloading the required `image_encoder` automatically on first load.
