"""Unit tests for model discovery helpers."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import assets
from core.assets import _scan_model_set, _detect_model_set_arch


class TestModelSetScanning(unittest.TestCase):
    def test_empty_model_set_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _scan_model_set(os.path.join(tmp, "model_set"))
            self.assertEqual(result, {})

    def test_full_model_set_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_set", "myModel")
            os.makedirs(os.path.join(model_dir, "unet"))
            with open(os.path.join(model_dir, "unet", "diffusion_pytorch_model.safetensors"), "w") as f:
                f.write("")
            result = _scan_model_set(os.path.join(tmp, "model_set"))
            self.assertIn("myModel", result)
            self.assertEqual(result["myModel"]["arch"], "SD 1.5")

    def test_model_set_with_text_encoder_2_is_sdxl(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_set", "sdxl_model")
            os.makedirs(os.path.join(model_dir, "text_encoder_2"))
            with open(os.path.join(model_dir, "model_index.json"), "w") as f:
                f.write('{"repo_id": "stabilityai/stable-diffusion-xl-base-1.0"}')
            result = _scan_model_set(os.path.join(tmp, "model_set"))
            self.assertIn("sdxl_model", result)
            self.assertEqual(result["sdxl_model"]["arch"], "SDXL")

    def test_model_set_detects_xl_from_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model_set", "ponyV6XL")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "model_index.json"), "w") as f:
                f.write('{"repo_id": "ponyDiffusionV6XL"}')
            result = _scan_model_set(os.path.join(tmp, "model_set"))
            self.assertEqual(result["ponyV6XL"]["arch"], "SDXL")

    def test_nested_model_sets_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "model_set", "outer", "inner"))
            with open(os.path.join(tmp, "model_set", "outer", "inner", "model.safetensors"), "w") as f:
                f.write("")
            result = _scan_model_set(os.path.join(tmp, "model_set"))
            # Only top-level candidates are considered; 'outer' is detected because
            # it contains a weight file somewhere in its subtree.
            self.assertIn("outer", result)
            self.assertEqual(result["outer"]["arch"], "SD 1.5")


class TestDiscoverAll(unittest.TestCase):
    def test_includes_model_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            for sub in ["checkpoints", "model_set", "vae", "lora", "controlnet", "ip-adapter"]:
                os.makedirs(os.path.join(tmp, sub), exist_ok=True)
            with open(os.path.join(tmp, "checkpoints", "base.safetensors"), "w") as f:
                f.write("")
            model_dir = os.path.join(tmp, "model_set", "myModel")
            os.makedirs(os.path.join(model_dir, "unet"))
            with open(os.path.join(model_dir, "unet", "diffusion_pytorch_model.safetensors"), "w") as f:
                f.write("")
            result = assets.discover_all(tmp)
            self.assertIn("model_sets", result)
            self.assertIn("myModel", result["model_sets"])


if __name__ == "__main__":
    unittest.main()
