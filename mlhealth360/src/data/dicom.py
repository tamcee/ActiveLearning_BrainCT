"""Small DICOM utilities. The current training notebook uses NIfTI volumes."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .hu import dicom_pixels_to_hu

def read_dicom(path):
    """Read a DICOM file with pydicom and return (dataset, HU image)."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("Install pydicom to read DICOM: pip install pydicom") from exc
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array
    hu = dicom_pixels_to_hu(
        pixels,
        getattr(ds, "RescaleSlope", 1.0),
        getattr(ds, "RescaleIntercept", 0.0),
    )
    # Pixel padding is not tissue. Keep its handling explicit for downstream work.
    padding = getattr(ds, "PixelPaddingValue", None)
    if padding is not None:
        hu = hu.astype(np.float32)
        hu[np.asarray(pixels) == padding] = -1000.0
    return ds, hu

def list_dicom_files(directory):
    """List files recursively; caller can validate which are actual DICOMs."""
    return sorted(p for p in Path(directory).rglob("*") if p.is_file())
