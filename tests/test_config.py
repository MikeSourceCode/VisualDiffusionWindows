"""Unit tests for the AppConfig configuration model."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AppConfig
from core.persistence import save_preset, load_preset, list_presets, delete_preset


class TestAppConfig(unittest.TestCase):
    def test_defaults_are_safe(self):
        cfg = AppConfig()
        self.assertEqual(cfg.checkpoint, "")
        self.assertEqual(cfg.architecture, "SDXL")
        self.assertEqual(cfg.loras, [])

    def test_lora_specs_filters_disabled(self):
        cfg = AppConfig(loras=[("a.safetensors", 0.0), ("b.safetensors", 0.6)])
        self.assertEqual(cfg.lora_specs(), [("b.safetensors", 0.6)])

    def test_lora_specs_empty_when_all_zero(self):
        cfg = AppConfig(loras=[("", 0.0), ("", 0.0)])
        self.assertEqual(cfg.lora_specs(), [])

    def test_lora_specs_excludes_unnamed_with_weight(self):
        # A blank filename must never be loaded, even if a weight slider is > 0.
        cfg = AppConfig(loras=[("", 0.6), ("b.safetensors", 0.5)])
        self.assertEqual(cfg.lora_specs(), [("b.safetensors", 0.5)])

    def test_preview_every_clamped(self):
        cfg = AppConfig(steps=5, preview_frequency=10)
        self.assertEqual(cfg.preview_every(cfg.steps), 5)

    def test_preview_every_minimum(self):
        cfg = AppConfig(steps=20, preview_frequency=0)
        self.assertEqual(cfg.preview_every(cfg.steps), 1)

    def test_seed_materializes_when_zero(self):
        cfg = AppConfig(seed=0)
        s = cfg.effective_seed()
        self.assertGreater(s, 0)

    def test_seed_preserved_when_set(self):
        cfg = AppConfig(seed=12345)
        self.assertEqual(cfg.effective_seed(), 12345)

    def test_roundtrip_via_dict(self):
        cfg = AppConfig(checkpoint="x.safetensors", loras=[("a.safetensors", 0.5)], steps=30, cfg_scale=8.0)
        restored = AppConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        self.assertEqual(restored.checkpoint, "x.safetensors")
        self.assertEqual(restored.loras, [("a.safetensors", 0.5)])
        self.assertEqual(restored.steps, 30)
        self.assertEqual(restored.cfg_scale, 8.0)

    def test_from_dict_ignores_unknown_keys(self):
        cfg = AppConfig.from_dict({"checkpoint": "y.safetensors", "bogus": 999})
        self.assertEqual(cfg.checkpoint, "y.safetensors")


class TestPresetPersistence(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_presets.db")
        if os.path.exists(self.db):
            os.remove(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.remove(self.db)

    def test_save_and_load_preset(self):
        cfg = AppConfig(checkpoint="z.safetensors", steps=25)
        self.assertTrue(save_preset("demo", cfg, db_path=self.db))
        loaded = load_preset("demo", db_path=self.db)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.checkpoint, "z.safetensors")
        self.assertEqual(loaded.steps, 25)

    def test_list_and_delete_preset(self):
        save_preset("a", AppConfig(), db_path=self.db)
        self.assertIn("a", [p["name"] for p in list_presets(db_path=self.db)])
        delete_preset("a", db_path=self.db)
        self.assertNotIn("a", [p["name"] for p in list_presets(db_path=self.db)])

    def test_load_missing_preset_returns_none(self):
        self.assertIsNone(load_preset("nope", db_path=self.db))


if __name__ == "__main__":
    unittest.main()
