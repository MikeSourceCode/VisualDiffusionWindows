"""Unit tests for model-set validation."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import _validate_model_set


class TestValidateModelSet(unittest.TestCase):
    def test_missing_folder_returns_error(self):
        self.assertEqual(
            _validate_model_set("/tmp/nonexistent_model_set_12345"),
            ["Model set folder missing: /tmp/nonexistent_model_set_12345"],
        )

    def test_empty_folder_missing_index_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                _validate_model_set(tmp),
                [
                    "Missing model_index.json — download may be incomplete",
                    "No weight files (.safetensors/.bin) found in expected subfolders "
                    "(unet/ or transformer/). The download is likely incomplete.",
                ],
            )

    def test_folder_with_weights_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "unet"))
            with open(os.path.join(tmp, "unet", "diffusion_pytorch_model.safetensors"), "w") as f:
                f.write("")
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                f.write('{"_class_name": "StableDiffusionXLPipeline"}')
            self.assertEqual(_validate_model_set(tmp), [])

    def test_folder_with_index_but_no_weights_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                f.write('{"_class_name": "StableDiffusionXLPipeline"}')
            self.assertEqual(
                _validate_model_set(tmp),
                [
                    "No weight files (.safetensors/.bin) found in expected subfolders "
                    "(unet/ or transformer/). The download is likely incomplete."
                ],
            )

    def test_folder_with_weights_in_transformer_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "transformer"))
            with open(os.path.join(tmp, "transformer", "diffusion_pytorch_model.safetensors"), "w") as f:
                f.write("")
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                f.write('{"_class_name": "StableDiffusionXLPipeline"}')
            self.assertEqual(_validate_model_set(tmp), [])


if __name__ == "__main__":
    unittest.main()
