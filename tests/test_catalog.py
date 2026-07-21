"""Unit tests for the model catalog and its presence detection."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.catalog import CATALOG, by_key, missing_entries, CatalogEntry


class TestCatalog(unittest.TestCase):
    def test_catalog_nonempty_and_unique_keys(self):
        keys = [e.key for e in CATALOG]
        self.assertTrue(keys)
        self.assertEqual(len(keys), len(set(keys)), "catalog keys must be unique")

    def test_by_key_found_and_missing(self):
        self.assertIsNotNone(by_key("sdxl_base"))
        self.assertIsNone(by_key("does_not_exist"))

    def test_entries_have_required_fields(self):
        for e in CATALOG:
            self.assertTrue(e.repo)
            self.assertIn(e.kind, {"checkpoint", "vae", "controlnet", "annotator"})
            self.assertTrue(e.size)
            self.assertTrue(e.desc)

    def test_file_entry_presence_on_empty_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = CatalogEntry(key="k", kind="checkpoint", repo="r",
                                 dest="models/checkpoints", desc="d", size="1GB",
                                 files=["x.safetensors"])
            self.assertFalse(entry.is_present(tmp))
            # Create the expected file -> present.
            d = os.path.join(tmp, "models", "checkpoints")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "x.safetensors"), "w").close()
            self.assertTrue(entry.is_present(tmp))

    def test_snapshot_entry_presence_detects_any_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = CatalogEntry(key="k", kind="controlnet", repo="r",
                                 dest="models/controlnet/x", desc="d", size="1GB",
                                 files=[])
            self.assertFalse(entry.is_present(tmp))
            d = os.path.join(tmp, "models", "controlnet", "x")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "diffusion_pytorch_model.safetensors"), "w").close()
            self.assertTrue(entry.is_present(tmp))

    def test_missing_entries_returns_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = missing_entries(tmp)
            # On an empty root, all file/snapshot entries are missing (annotator
            # depends on the HF cache, so it is excluded from this assertion).
            file_based = [e for e in CATALOG if e.kind != "annotator"]
            for e in file_based:
                self.assertIn(e, missing)


if __name__ == "__main__":
    unittest.main()
