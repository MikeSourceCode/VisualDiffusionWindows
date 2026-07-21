"""Unit tests for OpenPose JSON parsing and skeleton rendering."""

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose import (
    parse_openpose_json,
    render_openpose_skeleton,
    skeleton_from_json,
    POSE_PAIRS,
)


def _sample_person_pixels():
    # 18 COCO keypoints, pixel coords, all confident.
    return [
        416, 200, 1, 416, 350, 1, 340, 360, 1, 300, 480, 1, 280, 600, 1,
        492, 360, 1, 532, 480, 1, 552, 600, 1, 370, 650, 1, 360, 900, 1,
        355, 1150, 1, 462, 650, 1, 472, 900, 1, 477, 1150, 1,
        400, 185, 1, 432, 185, 1, 380, 195, 1, 452, 195, 1,
    ]


class TestParseOpenPoseJSON(unittest.TestCase):
    def test_parses_string_and_dict(self):
        data = {"people": [{"pose_keypoints_2d": _sample_person_pixels()}]}
        from_str = parse_openpose_json(json.dumps(data))
        from_dict = parse_openpose_json(data)
        self.assertEqual(len(from_str), 1)
        self.assertEqual(from_str, from_dict)
        self.assertEqual(len(from_str[0]), 18)

    def test_parses_bytes(self):
        data = {"people": [{"pose_keypoints_2d": _sample_person_pixels()}]}
        people = parse_openpose_json(json.dumps(data).encode("utf-8"))
        self.assertEqual(len(people[0]), 18)

    def test_low_confidence_becomes_none(self):
        kp = [100, 100, 0.0, 200, 200, 1.0]
        people = parse_openpose_json({"people": [{"pose_keypoints_2d": kp}]})
        self.assertIsNone(people[0][0])
        self.assertIsNotNone(people[0][1])

    def test_empty_people(self):
        self.assertEqual(parse_openpose_json({"people": []}), [])


class TestRenderSkeleton(unittest.TestCase):
    def test_renders_nonempty_canvas(self):
        data = {"people": [{"pose_keypoints_2d": _sample_person_pixels()}]}
        img = skeleton_from_json(json.dumps(data), 832, 1472)
        self.assertEqual(img.size, (832, 1472))
        self.assertGreater(np.array(img).max(), 0)

    def test_pixel_coords_render(self):
        people = parse_openpose_json({"people": [{"pose_keypoints_2d": _sample_person_pixels()}]})
        img = render_openpose_skeleton(people, 832, 1472, normalized=False)
        self.assertGreater(np.array(img).sum(), 0)

    def test_normalized_coords_render(self):
        kp = [0.5, 0.1, 1, 0.5, 0.3, 1, 0.4, 0.35, 1]
        img = skeleton_from_json({"people": [{"pose_keypoints_2d": kp}]}, 400, 700)
        self.assertGreater(np.array(img).max(), 0)

    def test_empty_people_blank_canvas(self):
        img = render_openpose_skeleton([], 100, 100)
        self.assertEqual(int(np.array(img).max()), 0)

    def test_pose_pairs_valid_indices(self):
        for a, b in POSE_PAIRS:
            self.assertTrue(0 <= a < 18 and 0 <= b < 18)


if __name__ == "__main__":
    unittest.main()
