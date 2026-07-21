# VisualDiffusion

A localized client for generative text-to-image / image-to-image workflows.
The engine loads models you provide and runs entirely under your control — no
models, weights, or checkpoints are bundled or distributed by this project.
See `models/README.md` for how model files are obtained, and `safety.py` for
the built-in content-filtering module.

## ⚖️ Ethical Use & Safety Disclaimer

This project is a localized user interface tool designed for generative text-to-image workflows.

* **No Data Bundling:** This software is an empty engine. It does not bundle, host, or distribute AI models, safety weights, or fine-tuned checkpoints. Users are entirely responsible for the models and services they choose to connect to this client.
* **Safe-by-Default:** By default, this application executes with its content-filter hooks enabled — including the output-side NSFW image guard and the non-Latin / CJK prompt refusal. The operator-supplied blocklist starts empty; disabling or weakening any of these mechanics requires manual developer configuration and places all generated-output compliance liability solely onto the operator.
* **Prohibited Content:** The developer strictly prohibits the use of this interface for generating illegal content, including Child Sexual Abuse Material (CSAM), non-consensual sexual imagery (Deepfakes), or material that violates local regulations. Note that face-swap / deepfake generation paths are present in the code but are disabled by default and gated by the prompt-side guard.

The developers provide this software as-is and are not liable for any output produced by it. The operator is solely responsible for complying with the laws and regulations that apply to them.

## 🛡️ Safety / Content Filtering

All moderation lives in the single importable module **`safety.py`**, which the
generation scripts call at two points: before generation (prompt gate) and
after each preview / final image (output guard). It is organized into three
ordered layers:

1. **`BLOCKED_LIST` (prompt term gate).** An operator-maintained list of terms
   to refuse (e.g. a competitor's name, a specific individual, or any term the
   operator chooses to exclude). Starts **empty** — you decide its contents.
   Prompts written in non-Latin scripts (CJK / Japanese / Korean) are refused
   regardless of the list, as this tool is English / Latin-prompt only.
2. **Text Classifier Trigger Abort (enabled by default; operator can opt-out).** When enabled, a text classifier
   you have selected aborts generation if it flags the positive prompt. The
   project does **not** bundle or recommend any specific classifier; you choose
   one that aligns with the laws and regulations applicable to you and set
   `TEXT_CLASSIFIER_MODEL` accordingly. You can disable this during setup or
   in `config/app_config.json`.
3. **Preview / Final Image Guard (CompVis safety checker, via diffusers).** After
   an image is generated (and for each decoded preview latent), it is inspected
   by the `CompVis/stable-diffusion-safety-checker` through diffusers' own
   `StableDiffusionSafetyChecker` class. Any flagged image is replaced with a
   black image, and generation aborts early if a preview is flagged. This acts
   as a guard against likely and potential NSFW generated outputs. See
   `models/safety_checker/README.md` for the model, its source, and its scope
   and limits.

**Operator responsibility.** These are technical guards, not a substitute for
legal compliance or human judgment. Disabling or weakening any layer requires
manual configuration and places all generated-output compliance liability onto
the operator. The operator remains responsible for every prompt submitted and
every image produced.

