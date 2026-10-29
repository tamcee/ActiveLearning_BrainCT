"""Three-window CT representation used by the 2.5D U-Net."""
from __future__ import annotations
import numpy as np
from .hu import hu_to_window

WINDOWS = (
    ("brain", 40, 80),
    ("subdural", 75, 215),
    ("bone", 400, 1800),
)

def window_slice(hu_slice, windows=WINDOWS):
    """Return float32 array shaped (3, H, W), values in [0, 1]."""
    channels = [hu_to_window(hu_slice, level, width) for _, level, width in windows]
    result = np.stack(channels, axis=0).astype(np.float32, copy=False)
    if result.ndim != 3 or result.shape[0] != len(windows):
        raise ValueError(f"Unexpected windowed shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite values after windowing")
    return result
