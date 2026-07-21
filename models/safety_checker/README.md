# `models/safety_checker/`

Output-side image-safety filter for VisualDiffusion (used by
`image_to_image_diffusion.py`, `img_ip_adapter_test.py`, and the censored fork
via the shared `safety` module).

Unlike the prompt-side moderation gate (Layer 1 `BLOCKED_LIST` + Layer 2 opt-in
text classifier), this is an **OUTPUT guard**: after an image is generated (and
for each preview latent), it is passed through the
`CompVis/stable-diffusion-safety-checker` CLIP model, and any image flagged is
replaced with a black image. This requires NO forbidden terms in the source
code — the concepts live inside the model weights.

**What it does:** acts as a guard against likely and potential NSFW (adult
sexual / nude) generated outputs, by inspecting the produced image and
blacking out anything the checker flags.

The operator remains responsible for all outputs and for complying with the
laws and regulations that apply to them. Read the model card / README (links
below) for the checker's actual scope and limits — it is a general-purpose
image filter, and its coverage is defined by the model, not by this project.

## Model

- **Source:** `CompVis/stable-diffusion-safety-checker` (Hugging Face)
- **Provided by:** CompVis — the group that created Stable Diffusion. Shipped
  as part of the `diffusers` library (`diffusers.pipelines.stable_diffusion.
  safety_checker.StableDiffusionSafetyChecker`).
- **Architecture:** CLIP vision model + concept-embedding classifier.
- **License & scope:** see the model card / README below.

### Links

- Repo:  https://huggingface.co/CompVis/stable-diffusion-safety-checker
- Readme: https://huggingface.co/CompVis/stable-diffusion-safety-checker/blob/main/README.md

## Download

The app auto-detects a local copy here first, then falls back to pulling from
the Hugging Face Hub at runtime (cached in `~/.cache/huggingface/hub`).

To vendor it locally so the app never hits the network for this model, run
from inside this folder:

```bash
curl -L -o config.json "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/config.json?download=true"
curl -L -o model.safetensors "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/model.safetensors?download=true"
curl -L -o preprocessor_config.json "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/preprocessor_config.json?download=true"
curl -L -o pytorch_model.bin "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/pytorch_model.bin?download=true"
curl -L -o safety_checker.py "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/safety_checker.py?download=true"
```

After download the folder should look like:

```
models/safety_checker/
  README.md
  config.json
  model.safetensors
  preprocessor_config.json
  pytorch_model.bin
  safety_checker.py
```

## Notes

- These files are **not committed to git** (see repo `.gitignore`); each user
  downloads them on their own accord.
- If the local files are missing or deleted, the app transparently falls back
  to the Hugging Face Hub repo id.
- The **Safety Checker** is controlled in `safety.py` by a single code-level
  constant (deliberately NOT a UI toggle):
  - `SAFETY_CHECKER_BLACK_OUT_NSFW` — equivalent of Fooocus's
    `default_black_out_nsfw` ("Black Out NSFW"). `True` (default) runs the
    checker and replaces flagged images with a black image; `False` turns the
    checker OFF entirely (images pass through unchanged). To disable it the
    operator must edit `safety.py` — there is no warn-only mode and no second
    flag.
