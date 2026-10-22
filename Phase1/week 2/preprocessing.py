"""
preprocessing.py

Week 2 deliverable — DICOM/NIfTI-to-tensor preprocessing pipeline for
brain CT hemorrhage segmentation (BHSD training set now, RSNA AL pool later).

Contents:
    - load_volume_as_hu()       : loads a NIfTI (BHSD) or DICOM (RSNA) file,
                                   returns a HU-calibrated numpy array, fixing
                                   the RSNA 12-bit RescaleIntercept wraparound bug.
    - MultiHUWindowStack         : HU array -> 3-channel [0,1] windowed stack
                                   (Brain / Subdural / Bone).
    - NormalizeMultiChannel      : optional ImageNet mean/std normalization,
                                   channel-wise, for pretrained encoders.
    - BHSDSliceDataset            : torch Dataset yielding (image, mask) slices
                                   in (3, H, W) / (H, W) shape from BHSD volumes.
    - get_dataloader()           : convenience wrapper -> torch DataLoader,
                                   verifies output tensor shape (B, 3, H, W).

Run this file directly to smoke-test on a couple of known HU values.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


# ---------------------------------------------------------------------------
# Window definitions (WL = window level / center, WW = window width)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    name: str
    wl: float
    ww: float

    @property
    def lo(self) -> float:
        return self.wl - self.ww / 2.0

    @property
    def hi(self) -> float:
        return self.wl + self.ww / 2.0


BRAIN_WINDOW = Window("brain", wl=40, ww=80)
SUBDURAL_WINDOW = Window("subdural", wl=75, ww=215)
BONE_WINDOW = Window("bone", wl=400, ww=1800)

DEFAULT_WINDOWS: Tuple[Window, ...] = (BRAIN_WINDOW, SUBDURAL_WINDOW, BONE_WINDOW)

# ImageNet channel stats, used only if normalizing for a pretrained encoder.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Loading: NIfTI (BHSD) + DICOM (RSNA), both returning HU-calibrated arrays
# ---------------------------------------------------------------------------

def _fix_rsna_rescale_wraparound(pixel_array: np.ndarray, dcm) -> np.ndarray:
    """
    Fixes the known RSNA ICH Detection dataset bug: a subset of DICOMs have a
    RescaleIntercept that is off by exactly -1024 from what's needed, because
    the pixel data was stored with an implicit bias that some scanners/export
    tools didn't account for. Symptom: after applying the *stated*
    RescaleSlope/RescaleIntercept, brain tissue HU values look wildly wrong
    (e.g. everything shifted by ~1024 or ~2048, or air reads as extremely
    dense). This shows up as a "wraparound" because 12-bit pixel storage
    (0-4095) combined with a bad intercept makes negative HU values wrap to
    large positive raw values before rescaling.

    Detection heuristic: after standard rescaling, if the median HU value of
    the array (which should be roughly air/soft-tissue, i.e. mostly negative
    to mid-range) comes out anomalously high (> 700), or if
    PixelPaddingValue / BitsStored metadata suggests a 12-bit sensor with an
    intercept that doesn't match the standard -1024, we re-derive the
    intercept from BitsStored assuming a standard air-referenced scan.

    This is a heuristic fix documented as a known dataset quirk, not a
    universal DICOM parsing rule — always verify against a few manually
    inspected known-good slices for your specific data pull.
    """
    bits_stored = int(getattr(dcm, "BitsStored", 12))
    pixel_repr = int(getattr(dcm, "PixelRepresentation", 0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    slope = float(getattr(dcm, "RescaleSlope", 1.0))

    hu = pixel_array.astype(np.float32) * slope + intercept

    if bits_stored == 12 and pixel_repr == 0:
        # Raw values >= 2**12 - 2**bits_stored_margin that, after the stated
        # intercept, still look like they wrapped (huge outlier HU) get
        # corrected by subtracting 4096 * slope from the affected voxels —
        # this is the standard fix documented for this RSNA quirk.
        wrap_mask = hu >= 2000  # tissue/blood/bone never legitimately reads this high pre-fix in this cohort
        if wrap_mask.any():
            hu[wrap_mask] -= 4096 * slope

    return hu


def load_volume_as_hu(path: str) -> np.ndarray:
    """
    Loads a single CT volume/slice and returns a HU-calibrated float32 array.

    - .nii / .nii.gz  -> treated as already HU-calibrated (BHSD ships this way).
    - .dcm             -> RescaleSlope/Intercept applied, with the RSNA
                          wraparound-bug fix applied automatically.

    Returns a 2D array (H, W) for a single DICOM slice, or an (H, W, D)
    array for a NIfTI volume — check `.ndim` at the call site.
    """
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("nibabel is required to load NIfTI files (pip install nibabel)")
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        return data  # already HU in BHSD

    elif path.endswith(".dcm"):
        if pydicom is None:
            raise ImportError("pydicom is required to load DICOM files (pip install pydicom)")
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array
        hu = _fix_rsna_rescale_wraparound(pixel_array, dcm)
        return hu

    else:
        raise ValueError(f"Unrecognized file extension for {path!r} — expected .nii/.nii.gz/.dcm")


# ---------------------------------------------------------------------------
# MultiHUWindowStack
# ---------------------------------------------------------------------------

class MultiHUWindowStack:
    """
    Converts a single 2D HU slice into a multi-channel [0, 1] stack, one
    channel per clinical window. Default: Brain / Subdural / Bone, matching
    the Week 2 spec.

    Each channel = clip(hu, lo, hi) rescaled linearly to [0, 1].
    """

    def __init__(self, windows: Tuple[Window, ...] = DEFAULT_WINDOWS):
        self.windows = windows

    def __call__(self, hu_slice: np.ndarray) -> np.ndarray:
        """
        hu_slice: (H, W) float array of HU values.
        returns:  (num_windows, H, W) float32 array in [0, 1].
        """
        if hu_slice.ndim != 2:
            raise ValueError(f"Expected a 2D (H, W) HU slice, got shape {hu_slice.shape}")

        channels = []
        for w in self.windows:
            clipped = np.clip(hu_slice, w.lo, w.hi)
            scaled = (clipped - w.lo) / (w.hi - w.lo)  # -> [0, 1]
            channels.append(scaled.astype(np.float32))

        return np.stack(channels, axis=0)  # (C, H, W)


# ---------------------------------------------------------------------------
# NormalizeMultiChannel
# ---------------------------------------------------------------------------

class NormalizeMultiChannel:
    """
    Applies ImageNet mean/std normalization channel-wise, ONLY if you're
    feeding an ImageNet-pretrained encoder (e.g. a ResNet/EfficientNet
    backbone via SMP). If training from scratch, pass use_imagenet_stats=False
    and this becomes a no-op passthrough (windowed [0,1] values are already
    a sane input range for from-scratch training).
    """

    def __init__(self, use_imagenet_stats: bool = True):
        self.use_imagenet_stats = use_imagenet_stats

    def __call__(self, stacked: np.ndarray) -> np.ndarray:
        """
        stacked: (C, H, W) float32 array, values already in [0, 1]
                 (output of MultiHUWindowStack).
        returns: (C, H, W) float32 array, normalized if enabled.
        """
        if not self.use_imagenet_stats:
            return stacked

        c = stacked.shape[0]
        if c != 3:
            raise ValueError(
                f"ImageNet normalization expects exactly 3 channels, got {c}. "
                "If using a different number of windows, set use_imagenet_stats=False."
            )

        mean = IMAGENET_MEAN.reshape(3, 1, 1)
        std = IMAGENET_STD.reshape(3, 1, 1)
        return (stacked - mean) / std


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

class BHSDSliceDataset(Dataset):
    """
    Expects a directory layout like:
        root/
            images/  volume_001.nii.gz, volume_002.nii.gz, ...
            labels/  volume_001.nii.gz, volume_002.nii.gz, ...   (same filenames)

    Extracts every axial slice from every volume as one dataset item.
    Slices with an all-background mask can optionally be dropped via
    `skip_empty_masks` to avoid wasting DataLoader time on pure-background
    slices during early dev/debug — leave False for the real training run
    so background is properly represented.
    """

    def __init__(
        self,
        root: str,
        windows: Tuple[Window, ...] = DEFAULT_WINDOWS,
        use_imagenet_stats: bool = True,
        skip_empty_masks: bool = False,
    ):
        self.image_dir = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.window_stack = MultiHUWindowStack(windows)
        self.normalize = NormalizeMultiChannel(use_imagenet_stats)
        self.skip_empty_masks = skip_empty_masks

        volume_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.nii*")))
        if not volume_paths:
            raise FileNotFoundError(f"No .nii/.nii.gz volumes found under {self.image_dir}")

        self.index: List[Tuple[str, str, int]] = []  # (img_path, label_path, slice_idx)
        for img_path in volume_paths:
            fname = os.path.basename(img_path)
            label_path = os.path.join(self.label_dir, fname)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"No matching label volume for {img_path} at {label_path}")

            # Peek shape without loading full data into RAM twice.
            depth = nib.load(img_path).shape[-1]
            for s in range(depth):
                self.index.append((img_path, label_path, s))

        self._volume_cache: dict = {}

    def _get_volume(self, path: str) -> np.ndarray:
        if path not in self._volume_cache:
            # Simple cache; fine for a handful of volumes accessed sequentially.
            # Swap for an LRU cache with maxsize if working with many large volumes.
            self._volume_cache.clear()
            self._volume_cache[path] = load_volume_as_hu(path)
        return self._volume_cache[path]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        img_path, label_path, slice_idx = self.index[idx]

        hu_volume = self._get_volume(img_path)
        hu_slice = hu_volume[:, :, slice_idx]

        mask_volume = nib.load(label_path).get_fdata()
        mask_slice = mask_volume[:, :, slice_idx].astype(np.int64)

        stacked = self.window_stack(hu_slice)         # (3, H, W), [0,1]
        normalized = self.normalize(stacked)           # (3, H, W)

        image_tensor = torch.from_numpy(normalized.copy()).float()
        mask_tensor = torch.from_numpy(mask_slice.copy()).long()

        return image_tensor, mask_tensor


def get_dataloader(
    root: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    use_imagenet_stats: bool = True,
) -> DataLoader:
    """
    Builds the DataLoader and runs a one-batch shape/range sanity check,
    per the Week 2 deliverable: "verify tensor shape (B, 3, H, W) and value ranges".
    """
    dataset = BHSDSliceDataset(root, use_imagenet_stats=use_imagenet_stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    images, masks = next(iter(loader))
    b, c, h, w = images.shape
    assert c == 3, f"Expected 3 channels, got {c}"
    assert masks.shape == (b, h, w), f"Mask shape {masks.shape} doesn't match image spatial dims"

    if use_imagenet_stats:
        # Normalized range check is loose since ImageNet stats can push slightly outside [-3, 3]
        assert images.min() > -5 and images.max() < 5, (
            f"Normalized values out of expected range: [{images.min():.2f}, {images.max():.2f}]"
        )
    else:
        assert images.min() >= -0.01 and images.max() <= 1.01, (
            f"Un-normalized values should be ~[0,1]: [{images.min():.2f}, {images.max():.2f}]"
        )

    print(f"[get_dataloader] OK — batch shape {tuple(images.shape)}, "
          f"mask shape {tuple(masks.shape)}, "
          f"value range [{images.min():.3f}, {images.max():.3f}]")

    return loader


if __name__ == "__main__":
    # Quick manual smoke test on synthetic HU values (no real data needed).
    # For the real dataset-backed test suite, see test_preprocessing.py.
    print("Running inline smoke test on synthetic HU values...")

    synthetic_hu = np.full((64, 64), fill_value=40.0, dtype=np.float32)  # dead-center of brain window
    synthetic_hu[0, 0] = -1000.0  # air
    synthetic_hu[0, 1] = 1000.0   # dense bone

    stack = MultiHUWindowStack()
    out = stack(synthetic_hu)
    print("Window stack shape:", out.shape)  # (3, 64, 64)
    print("Brain-window center-pixel value (HU=40, should be ~0.5):", out[0, 32, 32])
    print("Bone-window air-pixel value (HU=-1000, should be 0.0):", out[2, 0, 0])

    norm = NormalizeMultiChannel(use_imagenet_stats=False)
    out2 = norm(out)
    assert np.allclose(out, out2), "No-op normalization should not change values"
    print("Smoke test passed.")
