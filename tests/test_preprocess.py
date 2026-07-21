"""Unit tests for control-map preprocessors (PyraCanny / Canny)."""

import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocess import pyracanny, canny, preprocess


def _edged_image(w=160, h=240):
    """An image with a clear rectangle edge so Canny has something to find."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255
    return Image.fromarray(arr)


class TestPreprocessors(unittest.TestCase):
    def test_pyracanny_preserves_size(self):
        img = _edged_image()
        out = pyracanny(img, levels=3)
        self.assertEqual(out.size, img.size)
        self.assertEqual(out.mode, "RGB")

    def test_pyracanny_detects_edges(self):
        out = pyracanny(_edged_image(), levels=3)
        self.assertGreater(np.array(out).max(), 0)

    def test_pyracanny_levels_clamped(self):
        # levels < 1 must not crash; treated as 1.
        out = pyracanny(_edged_image(), levels=0)
        self.assertEqual(out.size, _edged_image().size)

    def test_canny_preserves_size(self):
        img = _edged_image()
        out = canny(img)
        self.assertEqual(out.size, img.size)
        self.assertEqual(out.mode, "RGB")

    def test_preprocess_dispatch(self):
        img = _edged_image()
        self.assertEqual(preprocess(img, "PyraCanny", levels=2).size, img.size)
        self.assertEqual(preprocess(img, "Canny").size, img.size)

    def test_preprocess_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            preprocess(_edged_image(), "Nope")

    def test_pyracanny_more_levels_not_fewer_edges(self):
        img = _edged_image()
        one = np.array(pyracanny(img, levels=1)).sum()
        three = np.array(pyracanny(img, levels=3)).sum()
        # Merging more scales only adds edges (pixel-wise max), never removes.
        self.assertGreaterEqual(three, one)


if __name__ == "__main__":
    unittest.main()
