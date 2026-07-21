# `models/safety/`

Optional **text classifier** weights for VisualDiffusion's Layer 2 prompt gate
(`safety.py`).

This folder is where you place a text-classifier model you have selected. The
app does NOT bundle or recommend any specific classifier — you choose one that
aligns with the laws and regulations applicable to you and set
`TEXT_CLASSIFIER_MODEL` in `safety.py` to its repo id or a local path. The
primary gate (Layer 1 `BLOCKED_LIST` + Layer 3 image censor) needs no download.

When `TEXT_CLASSIFIER_ENABLED = True` in `safety.py`, the app looks here first,
then falls back to the Hugging Face Hub repo id you configured.

## Choosing a model

Pick a classifier from a source you trust and are authorized to use. Examples of
the kind of model this slot expects (operator's responsibility to vet):

- A reputable organization's moderation model (e.g. an org-published
  text-moderation checkpoint), or
- Any Hugging Face `text-classification` model whose labels include an
  `nsfw` / `flagged` / `unsafe` score.

## Download (example shape only)

The exact filenames depend on the model you choose. As an example, for a
Diffusers-format checkpoint, run from inside this folder:

```bash
curl -L -o config.json   "https://huggingface.co/<your-chosen-repo>/resolve/main/config.json?download=true"
curl -L -o model.safetensors "https://huggingface.co/<your-chosen-repo>/resolve/main/model.safetensors?download=true"
curl -L -o tokenizer.json "https://huggingface.co/<your-chosen-repo>/resolve/main/tokenizer.json?download=true"
curl -L -o tokenizer_config.json "https://huggingface.co/<your-chosen-repo>/resolve/main/tokenizer_config.json?download=true"
curl -L -o special_tokens_map.json "https://huggingface.co/<your-chosen-repo>/resolve/main/special_tokens_map.json?download=true"
curl -L -o vocab.txt "https://huggingface.co/<your-chosen-repo>/resolve/main/vocab.txt?download=true"
```

After download the folder should contain the model's files plus this README:

```
models/safety/
  README.md
  (model files for the classifier you selected)
```

## Notes

- These files are **not committed to git** (see repo `.gitignore`); each user
  downloads them on their own accord.
- If the local files are missing or deleted, the app falls back to the
  Hugging Face Hub repo id you set in `TEXT_CLASSIFIER_MODEL`.
- The classifier scans ONLY the positive prompt. Negative prompts may contain
  explicit steer-away terms (e.g. `naked` to avoid nudity) and are intentionally
  excluded from classifier scoring.
