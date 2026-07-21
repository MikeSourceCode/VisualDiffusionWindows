"""Tests for the tag<->prompt sync helpers used by prompt_with_tags in app.py.

These cover the pure fragment logic and the reconcile callback that injects
selected pill text into the positive-prompt box (and removes it on deselect)
while preserving the user's manual edits.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app


class _SS(dict):
    """Minimal st.session_state stand-in supporting attr + .get access."""
    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v


class TestFragmentHelpers(unittest.TestCase):
    def test_append(self):
        self.assertEqual(app._append_fragment("", "cyberpunk"), "cyberpunk")
        self.assertEqual(app._append_fragment("a cat", "cyberpunk"), "a cat, cyberpunk")
        self.assertEqual(app._append_fragment("a cat, ", "cyberpunk"), "a cat, cyberpunk")

    def test_contains(self):
        self.assertTrue(app._contains_fragment("a cat, cyberpunk", "cyberpunk"))
        self.assertFalse(app._contains_fragment("a cat", "cyberpunk"))
        # substring should not count as contained (comma-token match only)
        self.assertFalse(app._contains_fragment("a cyberpunkish cat", "cyberpunk"))

    def test_remove(self):
        self.assertEqual(app._remove_fragment("a cat, cyberpunk, dog", "cyberpunk"), "a cat, dog")
        self.assertEqual(app._remove_fragment("a cat, cyberpunk", "cyberpunk"), "a cat")
        self.assertEqual(app._remove_fragment("a cat", "cyberpunk"), "a cat")


class TestSyncTagsIntoPrompt(unittest.TestCase):
    def setUp(self):
        self._orig = app.st.session_state
        self.ss = _SS()
        app.st.session_state = self.ss
        self.map = {"C": "cyberpunk", "A": "anime style"}

    def tearDown(self):
        app.st.session_state = self._orig

    def _sync(self):
        app._sync_tags_into_prompt("p", "pills", "applied", self.map)

    def test_select_appends(self):
        self.ss["p"] = "a serene lake"
        self.ss["pills"] = ["C"]
        self._sync()
        self.assertEqual(self.ss["p"], "a serene lake, cyberpunk")
        self.assertEqual(self.ss["applied"], ["cyberpunk"])

    def test_add_second_then_remove_first(self):
        self.ss["p"] = "a serene lake"
        self.ss["pills"] = ["C"]; self._sync()
        self.ss["pills"] = ["C", "A"]; self._sync()
        self.assertEqual(self.ss["p"], "a serene lake, cyberpunk, anime style")
        self.ss["pills"] = ["A"]; self._sync()
        self.assertEqual(self.ss["p"], "a serene lake, anime style")

    def test_manual_edits_preserved_on_deselect(self):
        self.ss["p"] = "a serene lake"
        self.ss["pills"] = ["A"]; self._sync()
        # user manually edits the prompt
        self.ss["p"] = "a serene lake, anime style, extra words"
        self.ss["pills"] = []; self._sync()
        self.assertEqual(self.ss["p"], "a serene lake, extra words")

    def test_no_duplicate_when_already_present(self):
        self.ss["p"] = "masterpiece, cyberpunk"
        self.ss["pills"] = ["C"]
        self._sync()
        # cyberpunk already present -> not duplicated
        self.assertEqual(self.ss["p"], "masterpiece, cyberpunk")


if __name__ == "__main__":
    unittest.main()
