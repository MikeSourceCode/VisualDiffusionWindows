"""Tests for the live-preview generation path.

Live preview steps are decodes of the scheduler's predicted clean latent (no
per-step NSFW model pass — that would add a full CLIP forward pass every step
and ~3x the runtime versus the standalone scripts). The authoritative safety
check is the final-image output guard (safety.censor), which always runs.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.engine as eng


class TestEarlySafetyGate(unittest.TestCase):
    def _early_step(self, total_steps):
        # Mirrors core.engine.generate(): single check at ~20% progress.
        return max(1, round(total_steps * 0.2))

    def test_early_step_at_20_percent(self):
        self.assertEqual(self._early_step(8), 2)   # 1.6 -> 2
        self.assertEqual(self._early_step(16), 3)  # 3.2 -> 3
        self.assertEqual(self._early_step(20), 4)  # 4.0 -> 4
        self.assertEqual(self._early_step(10), 2)  # 2.0 -> 2

    def test_early_step_minimum_one(self):
        # Very short runs still check at step 1, never 0.
        self.assertEqual(self._early_step(1), 1)
        self.assertEqual(self._early_step(3), 1)


@unittest.skipUnless(
    os.environ.get("VD_RUN_MODEL_TESTS") == "1",
    "set VD_RUN_MODEL_TESTS=1 to run the slow model-backed generation test",
)
class TestPreviewIntegration(unittest.TestCase):
    """Opt-in: real SDXL generation must produce a non-flat image.

    No per-step safety model pass, so this must complete in roughly the same
    time as the standalone scripts.
    """

    def test_benign_prompt_completes(self):
        import numpy as np
        from core import (AppConfig, detect_backend, detect_vram_mb,
                          get_vram_state, load_base_pipeline, generate)

        ckpt = "models/checkpoints/sd_xl_base_1.0.safetensors"
        if not os.path.exists(ckpt):
            self.skipTest("SDXL base checkpoint not present")

        backend = detect_backend()
        vram = get_vram_state(backend, detect_vram_mb(backend))
        pipe = load_base_pipeline(ckpt, None, backend, "SDXL", img2img=True)
        cfg = AppConfig(checkpoint="x", steps=8, cfg_scale=6.0, seed=123,
                        strength=0.6, show_previews=True, preview_frequency=1)
        img = generate(pipe, backend, vram, cfg,
                       "a serene mountain lake at sunrise", "blurry",
                       init_image=None, preview_callback=lambda i, s: None)
        a = np.asarray(img)
        self.assertFalse(int(a.min()) == int(a.max()), "output is a flat image")


if __name__ == "__main__":
    unittest.main()
