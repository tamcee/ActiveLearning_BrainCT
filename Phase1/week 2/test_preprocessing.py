"""
test_preprocessing.py

Unit tests for preprocessing.py, per the Week 2 deliverable requirement:
"known HU values -> expected window output; check normalised range matches
encoder expectation".

Run with:  pytest test_preprocessing.py -v
"""

import numpy as np
import pytest

from preprocessing import (
    MultiHUWindowStack,
    NormalizeMultiChannel,
    BRAIN_WINDOW,
    SUBDURAL_WINDOW,
    BONE_WINDOW,
    Window,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ---------------------------------------------------------------------------
# Window boundary math
# ---------------------------------------------------------------------------

def test_window_lo_hi_brain():
    # WL=40, WW=80 -> range [0, 80]
    assert BRAIN_WINDOW.lo == 0
    assert BRAIN_WINDOW.hi == 80


def test_window_lo_hi_subdural():
    # WL=75, WW=215 -> range [-32.5, 182.5]
    assert SUBDURAL_WINDOW.lo == pytest.approx(-32.5)
    assert SUBDURAL_WINDOW.hi == pytest.approx(182.5)


def test_window_lo_hi_bone():
    # WL=400, WW=1800 -> range [-500, 1300]
    assert BONE_WINDOW.lo == -500
    assert BONE_WINDOW.hi == 1300


# ---------------------------------------------------------------------------
# MultiHUWindowStack: known HU value -> expected windowed output
# ---------------------------------------------------------------------------

class TestMultiHUWindowStack:

    def setup_method(self):
        self.stack = MultiHUWindowStack()

    def _single_value_slice(self, hu_value: float, size=8) -> np.ndarray:
        return np.full((size, size), fill_value=hu_value, dtype=np.float32)

    def test_output_shape(self):
        hu = self._single_value_slice(40.0)
        out = self.stack(hu)
        assert out.shape == (3, 8, 8)

    def test_rejects_non_2d_input(self):
        bad_input = np.zeros((3, 8, 8), dtype=np.float32)  # already 3D
        with pytest.raises(ValueError):
            self.stack(bad_input)

    @pytest.mark.parametrize("hu_value,expected", [
        (-1000, 0.0),   # air: far below brain window lo (0) -> clipped to 0
        (0, 0.0),       # brain window lo
        (40, 0.5),      # brain window center (WL)
        (80, 1.0),      # brain window hi
        (200, 1.0),     # bone-range value: far above brain window hi -> clipped to 1
    ])
    def test_brain_channel_known_values(self, hu_value, expected):
        hu = self._single_value_slice(hu_value)
        out = self.stack(hu)
        brain_channel = out[0]
        assert brain_channel[0, 0] == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("hu_value,expected", [
        (-32.5, 0.0),   # subdural window lo
        (75, 0.5),      # subdural window center (WL)
        (182.5, 1.0),   # subdural window hi
    ])
    def test_subdural_channel_known_values(self, hu_value, expected):
        hu = self._single_value_slice(hu_value)
        out = self.stack(hu)
        subdural_channel = out[1]
        assert subdural_channel[0, 0] == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("hu_value,expected", [
        (-500, 0.0),    # bone window lo
        (400, 0.5),     # bone window center (WL)
        (1300, 1.0),    # bone window hi
        (-1000, 0.0),   # air: below bone lo -> clipped to 0
    ])
    def test_bone_channel_known_values(self, hu_value, expected):
        hu = self._single_value_slice(hu_value)
        out = self.stack(hu)
        bone_channel = out[2]
        assert bone_channel[0, 0] == pytest.approx(expected, abs=1e-5)

    def test_output_always_in_unit_range(self):
        # Sweep a wide range of HU values, output must always land in [0, 1]
        for hu_value in [-2000, -1000, -500, -100, 0, 40, 100, 400, 1000, 3000]:
            hu = self._single_value_slice(hu_value)
            out = self.stack(hu)
            assert out.min() >= 0.0
            assert out.max() <= 1.0

    def test_custom_window_subset(self):
        # Verify the class works with a non-default window tuple too
        custom = (Window("test", wl=0, ww=100),)  # range [-50, 50]
        stack = MultiHUWindowStack(windows=custom)
        hu = self._single_value_slice(0.0)
        out = stack(hu)
        assert out.shape == (1, 8, 8)
        assert out[0, 0, 0] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# NormalizeMultiChannel: normalized range matches encoder expectation
# ---------------------------------------------------------------------------

class TestNormalizeMultiChannel:

    def test_noop_when_disabled(self):
        norm = NormalizeMultiChannel(use_imagenet_stats=False)
        x = np.random.rand(3, 4, 4).astype(np.float32)
        out = norm(x)
        assert np.array_equal(x, out)

    def test_imagenet_normalization_known_values(self):
        norm = NormalizeMultiChannel(use_imagenet_stats=True)
        # Feed exactly the ImageNet mean on every channel -> output should be ~0
        x = np.zeros((3, 4, 4), dtype=np.float32)
        for c in range(3):
            x[c] = IMAGENET_MEAN[c]
        out = norm(x)
        assert np.allclose(out, 0.0, atol=1e-5)

    def test_imagenet_normalization_matches_formula(self):
        norm = NormalizeMultiChannel(use_imagenet_stats=True)
        x = np.full((3, 2, 2), 1.0, dtype=np.float32)  # all channels = 1.0 (max windowed value)
        out = norm(x)
        for c in range(3):
            expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            assert out[c, 0, 0] == pytest.approx(expected, abs=1e-5)

    def test_rejects_wrong_channel_count(self):
        norm = NormalizeMultiChannel(use_imagenet_stats=True)
        x = np.zeros((5, 4, 4), dtype=np.float32)  # 5 channels, not 3
        with pytest.raises(ValueError):
            norm(x)

    def test_output_range_reasonable_for_pretrained_encoder(self):
        # Windowed input is always in [0,1] -> normalized output should stay
        # within a few standard deviations either side of 0, never blow up.
        norm = NormalizeMultiChannel(use_imagenet_stats=True)
        x = np.random.rand(3, 16, 16).astype(np.float32)  # simulates windowed [0,1] input
        out = norm(x)
        assert out.min() > -5.0
        assert out.max() < 5.0


# ---------------------------------------------------------------------------
# Integration: window stack -> normalize, end to end
# ---------------------------------------------------------------------------

def test_full_pipeline_known_hu_to_normalized_output():
    """
    A single, fully-worked example: HU=40 (brain window center) through the
    full pipeline (window -> normalize) should land at a known value.
    """
    hu = np.full((4, 4), 40.0, dtype=np.float32)  # brain window center

    stack = MultiHUWindowStack()
    normalize = NormalizeMultiChannel(use_imagenet_stats=True)

    windowed = stack(hu)
    result = normalize(windowed)

    # Channel 0 (brain window): HU=40 -> windowed 0.5 -> normalized:
    expected_brain = (0.5 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert result[0, 0, 0] == pytest.approx(expected_brain, abs=1e-5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
