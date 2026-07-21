"""Unit tests for the click-to-prompt tag palette loader."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tags import load_tag_groups, tags_to_prompt_fragment


class TestTagLoader(unittest.TestCase):
    def test_loads_bundled_tags(self):
        groups = load_tag_groups()
        names = [g["name"] for g in groups]
        self.assertIn("Quality", names)
        self.assertIn("Scene", names)

    def test_bundled_tags_have_required_fields(self):
        for group in load_tag_groups():
            for tag in group["tags"]:
                self.assertIn("emoji", tag)
                self.assertIn("label", tag)
                self.assertIn("text", tag)
                self.assertTrue(tag["text"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_tag_groups("/nonexistent/tags.json"), [])

    def test_malformed_file_returns_empty(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            path = f.name
        try:
            self.assertEqual(load_tag_groups(path), [])
        finally:
            os.remove(path)

    def test_text_defaults_to_label(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"groups": [{"name": "G", "tags": [{"label": "Foo"}]}]}, f)
            path = f.name
        try:
            groups = load_tag_groups(path)
            self.assertEqual(groups[0]["tags"][0]["text"], "Foo")
        finally:
            os.remove(path)

    def test_group_without_name_skipped(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"groups": [{"tags": [{"label": "X"}]}]}, f)
            path = f.name
        try:
            self.assertEqual(load_tag_groups(path), [])
        finally:
            os.remove(path)


class TestPromptFragment(unittest.TestCase):
    def test_joins_with_commas(self):
        self.assertEqual(tags_to_prompt_fragment(["a", "b", "c"]), "a, b, c")

    def test_dedupes_case_insensitive_preserving_order(self):
        self.assertEqual(
            tags_to_prompt_fragment(["masterpiece", "Beach", "MASTERPIECE", "best quality"]),
            "masterpiece, Beach, best quality",
        )

    def test_empty_and_whitespace_filtered(self):
        self.assertEqual(tags_to_prompt_fragment(["", "  ", "cat"]), "cat")

    def test_empty_input(self):
        self.assertEqual(tags_to_prompt_fragment([]), "")


if __name__ == "__main__":
    unittest.main()
