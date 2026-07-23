"""Unit tests for combined checkpoint/model-set selection logic."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config_model_dirs


class TestCombinedCheckpoints(unittest.TestCase):
    def test_model_set_dir_in_config(self):
        dirs = config_model_dirs("/tmp/models")
        self.assertIn("model_set", dirs)
        self.assertEqual(dirs["model_set"], "/tmp/models/model_set")

    def test_config_dirs_include_all_required(self):
        dirs = config_model_dirs()
        expected = {"models", "checkpoints", "model_set", "vae", "lora", "controlnet", "ip_adapter"}
        self.assertTrue(expected.issubset(set(dirs.keys())))


if __name__ == "__main__":
    unittest.main()
