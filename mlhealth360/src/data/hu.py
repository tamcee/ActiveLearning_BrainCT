"""HU conversion helpers for DICOM and already-converted NIfTI volumes."""
from __future__ import annotations
import numpy as np

def dicom_pixels_to_hu(pixel_array, rescale_slope=1.0, rescale_intercept=0.0):
    """Convert stored DICOM pixel values to Hounsfield Units."""
    pixels = np.asarray(pixel_array, dtype=np.float32)
    return pixels * float(rescale_slope) + float(rescale_intercept)

def hu_to_window(image_hu, level: float, width: float):
    """Clip HU to a CT window and scale to [0, 1]."""
    if width <= 0:
        raise ValueError("Window width must be > 0")
    low = float(level) - float(width) / 2.0
    high = float(level) + float(width) / 2.0
    x = np.asarray(image_hu, dtype=np.float32)
    return np.clip((x - low) / (high - low), 0.0, 1.0).astype(np.float32)

def nifti_values_to_hu(volume, slope=1.0, intercept=0.0):
    """
    Convert NIfTI stored values to HU only when the dataset documentation
    says values are stored CT values needing slope/intercept. Nibabel's
    dataobj may already apply NIfTI scaling; do not apply this blindly.
    """
    return np.asarray(volume, dtype=np.float32) * float(slope) + float(intercept)
