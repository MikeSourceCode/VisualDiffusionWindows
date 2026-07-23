"""Unit tests for getmodel.py destination logic."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import getmodel


class TestGetModel(unittest.TestCase):
    def test_checkpoint_dir_defaults_to_models_checkpoints(self):
        expected = getmodel.REPO_ROOT / "models" / "checkpoints"
        self.assertEqual(getmodel.DEFAULT_CHECKPOINTS, expected)

    def test_model_set_dir_defaults_to_models_model_set(self):
        expected = getmodel.REPO_ROOT / "models" / "model_set"
        self.assertEqual(getmodel.DEFAULT_MODEL_SET, expected)

    def test_checkpoint_dir_exists(self):
        self.assertTrue(getmodel.DEFAULT_CHECKPOINTS.exists())

    def test_model_set_dir_exists(self):
        self.assertTrue(getmodel.DEFAULT_MODEL_SET.exists())


if __name__ == "__main__":
    unittest.main()
