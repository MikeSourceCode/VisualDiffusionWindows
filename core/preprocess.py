"""Control-map preprocessors for ControlNet-guided generation.

PyraCanny: a pyramid (multi-scale) Canny edge detector, following the Fooocus
approach. Plain single-pass Canny at one resolution and fixed thresholds either
misses large-scale structure or drowns in fine texture. Running Canny across a
downscaled image pyramid and merging the results yields cleaner, hierarchical
edges (coarse levels = composition/silhouette; fine levels = detail) that
ControlNet follows better for structure-preserving generation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _to_gray_np(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def canny(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Standard single-pass Canny edge map as a 3-channel RGB image."""
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv (cv2) is required for edge detection")
    edges = cv2.Canny(_to_gray_np(img), low, high)
    return Image.fromarray(edges).convert("RGB")


def pyracanny(img: Image.Image, low: int = 64, high: int = 128,
              levels: int = 3, blur: int = 0) -> Image.Image:
    """Pyramid multi-scale Canny.

    Runs Canny at ``levels`` progressively downscaled resolutions, upsamples
    each edge map back to full size, and merges them with a pixel-wise max so
    strong edges at any scale survive.

    Args:
        low, high: Canny hysteresis thresholds.
        levels: number of pyramid levels (>=1). Level 0 is full resolution.
        blur: optional odd-kernel Gaussian blur applied before edges to
              suppress noise (0 = no blur).
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv (cv2) is required for PyraCanny")
    levels = max(1, int(levels))
    gray = _to_gray_np(img)
    h, w = gray.shape[:2]
    if blur and blur >= 3:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    accum = np.zeros((h, w), dtype=np.uint8)
    for i in range(levels):
        scale = 1.0 / (2 ** i)
        lw = max(8, int(round(w * scale)))
        lh = max(8, int(round(h * scale)))
        level_img = gray if i == 0 else cv2.resize(gray, (lw, lh), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(level_img, low, high)
        if edges.shape[:2] != (h, w):
            edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
        accum = np.maximum(accum, edges)

    return Image.fromarray(accum).convert("RGB")


PREPROCESSORS = {
    "PyraCanny": lambda img, **kw: pyracanny(img, **kw),
    "Canny": lambda img, **kw: canny(img, low=kw.get("low", 100), high=kw.get("high", 200)),
}


def preprocess(img: Image.Image, method: str = "PyraCanny", **kwargs) -> Image.Image:
    fn = PREPROCESSORS.get(method)
    if fn is None:
        raise ValueError(f"Unknown preprocessor '{method}'. Options: {list(PREPROCESSORS)}")
    return fn(img, **kwargs)
