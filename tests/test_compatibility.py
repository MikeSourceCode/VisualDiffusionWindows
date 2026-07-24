"""Unit tests for core.compatibility."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compatibility import (
    CompatibilityResult,
    check_backend_compatibility,
    check_model_index,
    check_model_set_structure,
    validate_local_model_set,
    validate_repo,
)


class TestCheckModelIndex(unittest.TestCase):
    def test_missing_file_returns_false(self):
        ok, reasons = check_model_index("/tmp/nonexistent_12345.json")
        self.assertFalse(ok)
        self.assertIn("model_index.json is missing", reasons)

    def test_invalid_json_returns_false(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name
        try:
            ok, reasons = check_model_index(path)
            self.assertFalse(ok)
            self.assertIn("not valid JSON", reasons[0])
        finally:
            os.unlink(path)

    def test_missing_class_name_returns_false(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path = f.name
        try:
            ok, reasons = check_model_index(path)
            self.assertFalse(ok)
            self.assertIn("_class_name", reasons[0])
        finally:
            os.unlink(path)

    def test_unsupported_pipeline_class_returns_false(self):
        idx = {"_class_name": "MageFlowPipeline"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(idx, f)
            path = f.name
        try:
            ok, reasons = check_model_index(path)
            self.assertFalse(ok)
            self.assertIn("Unsupported pipeline class", reasons[0])
        finally:
            os.unlink(path)

    def test_sdxl_pipeline_class_passes(self):
        idx = {"_class_name": "StableDiffusionXLPipeline"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(idx, f)
            path = f.name
        try:
            ok, reasons = check_model_index(path)
            self.assertTrue(ok)
            self.assertEqual(reasons, [])
        finally:
            os.unlink(path)

    def test_sd15_pipeline_class_passes(self):
        idx = {"_class_name": "StableDiffusionPipeline"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(idx, f)
            path = f.name
        try:
            ok, reasons = check_model_index(path)
            self.assertTrue(ok)
            self.assertEqual(reasons, [])
        finally:
            os.unlink(path)

    def test_non_clip_text_encoder_fails(self):
        idx = {
            "_class_name": "StableDiffusionXLPipeline",
            "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
            "tokenizer": ["transformers", "AutoProcessor"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(idx, f)
            path = f.name
        try:
            ok, reasons = check_model_index(path)
            self.assertFalse(ok)
            self.assertTrue(any("text encoder" in r.lower() for r in reasons))
        finally:
            os.unlink(path)

    def test_clip_text_encoder_passes(self):
        idx = {
            "_class_name": "StableDiffusionXLPipeline",
            "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(idx, f)
            path = f.name
        try:
            ok, reasons = check_model_index(path)
            self.assertTrue(ok)
        finally:
            os.unlink(path)


class TestCheckModelSetStructure(unittest.TestCase):
    def test_missing_folder(self):
        ok, reasons = check_model_set_structure("/tmp/nonexistent_model_set_xyz")
        self.assertFalse(ok)
        self.assertIn("folder missing", reasons[0])

    def test_empty_folder_missing_index_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, reasons = check_model_set_structure(tmp)
            self.assertFalse(ok)
            self.assertTrue(any("model_index.json" in r for r in reasons))

    def test_folder_with_weights_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "unet"))
            with open(os.path.join(tmp, "unet", "diffusion_pytorch_model.safetensors"), "w") as f:
                f.write("")
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "StableDiffusionXLPipeline"}, f)
            ok, reasons = check_model_set_structure(tmp)
            self.assertTrue(ok)
            self.assertEqual(reasons, [])

    def test_folder_with_weights_in_transformer_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "transformer"))
            with open(os.path.join(tmp, "transformer", "diffusion_pytorch_model.safetensors"), "w") as f:
                f.write("")
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "StableDiffusionXLPipeline"}, f)
            ok, reasons = check_model_set_structure(tmp)
            self.assertTrue(ok)

    def test_folder_with_weights_in_text_encoder_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "text_encoder"))
            with open(os.path.join(tmp, "text_encoder", "model.safetensors"), "w") as f:
                f.write("")
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "StableDiffusionXLPipeline"}, f)
            ok, reasons = check_model_set_structure(tmp)
            self.assertTrue(ok)


class TestCheckBackendCompatibility(unittest.TestCase):
    def test_mage_flow_on_mps_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "MageFlowPipeline"}, f)
            ok, reasons = check_backend_compatibility(tmp, "mps")
            self.assertFalse(ok)
            self.assertTrue(any("MageFlow" in r for r in reasons))

    def test_sdxl_on_mps_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "StableDiffusionXLPipeline"}, f)
            ok, reasons = check_backend_compatibility(tmp, "mps")
            self.assertTrue(ok)

    def test_cuda_always_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "MageFlowPipeline"}, f)
            ok, reasons = check_backend_compatibility(tmp, "cuda")
            self.assertTrue(ok)


class TestValidateLocalModelSet(unittest.TestCase):
    def test_compatible_model_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "unet"))
            with open(os.path.join(tmp, "unet", "diffusion_pytorch_model.safetensors"), "w") as f:
                f.write("")
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "StableDiffusionXLPipeline"}, f)
            result = validate_local_model_set(tmp)
            self.assertTrue(result)
            self.assertEqual(result.reasons, [])

    def test_incompatible_model_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model_index.json"), "w") as f:
                json.dump({"_class_name": "MageFlowPipeline"}, f)
            result = validate_local_model_set(tmp, backend="mps")
            self.assertFalse(result)
            self.assertTrue(result.reasons)


class TestCompatibilityResult(unittest.TestCase):
    def test_bool_true_when_compatible(self):
        r = CompatibilityResult(True, [])
        self.assertTrue(bool(r))

    def test_bool_false_when_incompatible(self):
        r = CompatibilityResult(False, ["reason"])
        self.assertFalse(bool(r))


class TestValidateRepo(unittest.TestCase):
    def test_mage_flow_repo_is_incompatible(self):
        result = validate_repo("microsoft/Mage-Flow")
        self.assertFalse(result)
        self.assertTrue(any("MageFlowPipeline" in r for r in result.reasons))

    def test_sdxl_repo_is_compatible(self):
        result = validate_repo("stabilityai/stable-diffusion-xl-base-1.0")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
