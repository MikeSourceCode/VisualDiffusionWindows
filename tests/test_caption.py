"""Unit tests for the image captioning module (fallback behavior)."""

import os
import sys
import unittest

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.caption as caption_mod


class TestCaptionFallback(unittest.TestCase):
    def setUp(self):
        # Snapshot and reset module cache so tests are isolated.
        self._saved = dict(caption_mod._cache)

    def tearDown(self):
        caption_mod._cache.clear()
        caption_mod._cache.update(self._saved)

    def test_returns_placeholder_when_model_unavailable(self):
        # Force the "failed to load" path without importing transformers.
        caption_mod._cache["model"] = None
        caption_mod._cache["proc"] = None
        caption_mod._cache["failed"] = True
        img = Image.new("RGB", (32, 32), (10, 20, 30))
        self.assertEqual(caption_mod.caption_image(img), "a detailed scene")

    def test_is_available_reflects_failed_flag(self):
        caption_mod._cache["model"] = None
        caption_mod._cache["failed"] = True
        self.assertFalse(caption_mod.is_available())


if __name__ == "__main__":
    unittest.main()
