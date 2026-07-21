"""OpenPose skeleton control maps.

Two input paths for the Pose Control tab:
1. Image -> detected skeleton, via controlnet_aux OpenposeDetector (needs the
   ~200MB annotator weights, downloaded on first use).
2. JSON keypoints -> rendered skeleton, using the OpenPose COCO-18 body format
   ({"people":[{"pose_keypoints_2d":[x,y,conf, ... 18 triplets]}]}).

Both produce an RGB skeleton image drawn with the canonical OpenPose limb
colors on black, which is what ControlNet-OpenPose models are trained on.
"""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# Canonical OpenPose COCO-18 limb connections (pairs of keypoint indices).
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# Canonical OpenPose colors (BGR order for cv2), one per limb/joint.
POSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

_detector_cache = {"model": None, "failed": False}


def detect_pose_from_image(img: Image.Image, detect_resolution: int = 512,
                           image_resolution: int = 1024,
                           include_hand: bool = False,
                           include_face: bool = False) -> Optional[Image.Image]:
    """Run OpenPose on an image and return the skeleton control map (RGB)."""
    if _detector_cache["failed"]:
        return None
    if _detector_cache["model"] is None:
        try:
            from controlnet_aux import OpenposeDetector
            _detector_cache["model"] = OpenposeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
        except Exception:
            _detector_cache["failed"] = True
            return None
    detector = _detector_cache["model"]
    result = detector(
        img.convert("RGB"),
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
        include_body=True,
        include_hand=include_hand,
        include_face=include_face,
        output_type="pil",
    )
    return result


def parse_openpose_json(data) -> List[List[Optional[tuple]]]:
    """Parse OpenPose JSON into a list of people, each a list of 18 (x,y,conf).

    Accepts a JSON string, bytes, or an already-parsed dict. Missing/low-conf
    keypoints become None.
    """
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    if isinstance(data, str):
        data = json.loads(data)

    people = data.get("people", []) if isinstance(data, dict) else []
    parsed = []
    for person in people:
        flat = person.get("pose_keypoints_2d", [])
        pts: List[Optional[tuple]] = []
        for i in range(0, len(flat), 3):
            x, y = flat[i], flat[i + 1]
            conf = flat[i + 2] if i + 2 < len(flat) else 1.0
            pts.append((float(x), float(y), float(conf)) if conf > 0.05 else None)
        parsed.append(pts)
    return parsed


def render_openpose_skeleton(people: List[List[Optional[tuple]]],
                             width: int, height: int,
                             normalized: bool = False) -> Image.Image:
    """Draw OpenPose skeletons onto a black RGB canvas.

    If ``normalized`` is True, keypoint x/y are treated as 0..1 fractions of the
    canvas; otherwise they are absolute pixel coordinates.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv (cv2) is required to render pose skeletons")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def to_px(pt):
        x, y, c = pt
        if normalized:
            x, y = x * width, y * height
        return int(round(x)), int(round(y))

    for person in people:
        # Limbs
        for pair_idx, (a, b) in enumerate(POSE_PAIRS):
            if a < len(person) and b < len(person) and person[a] and person[b]:
                pa, pb = to_px(person[a]), to_px(person[b])
                color = POSE_COLORS[pair_idx % len(POSE_COLORS)]
                cv2.line(canvas, pa, pb, color, 4, lineType=cv2.LINE_AA)
        # Joints
        for j_idx, pt in enumerate(person):
            if pt:
                cv2.circle(canvas, to_px(pt), 4, POSE_COLORS[j_idx % len(POSE_COLORS)], -1)

    # cv2 uses BGR; convert to RGB for PIL.
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def skeleton_from_json(data, width: int = 832, height: int = 1472) -> Image.Image:
    """Convenience: JSON -> rendered OpenPose skeleton control map.

    Auto-detects normalized (0..1) vs pixel coordinates from the value range.
    """
    people = parse_openpose_json(data)
    max_val = 0.0
    for person in people:
        for pt in person:
            if pt:
                max_val = max(max_val, abs(pt[0]), abs(pt[1]))
    normalized = max_val <= 1.5  # coords all within [0,1] => normalized
    return render_openpose_skeleton(people, width, height, normalized=normalized)
