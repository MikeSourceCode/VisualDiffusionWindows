"""Unit tests for getmodel.py stub behavior."""

import os
import sys
import unittest
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import getmodel


class TestGetModelStub(unittest.TestCase):
    def test_main_prints_local_only_message(self):
        captured = StringIO()
        sys.stdout = captured
        try:
            with self.assertRaises(SystemExit) as cm:
                getmodel.main()
            self.assertEqual(cm.exception.code, 0)
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("no longer downloads models", output.lower())
        self.assertIn("models/checkpoints/", output)
        self.assertIn("models/model_set/", output)


if __name__ == "__main__":
    unittest.main()
