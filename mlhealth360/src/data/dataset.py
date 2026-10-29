"""Study-aware 2.5D dataset for paired NIfTI CT volumes and masks."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from .windowing import window_slice

def pair_nifti_files(image_dir, mask_dir, suffix=".nii.gz"):
    """Pair files by basename. Images and masks must share filenames."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    images = {p.name: p for p in image_dir.glob(f"*{suffix}")}
    masks = {p.name: p for p in mask_dir.glob(f"*{suffix}")}
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise FileNotFoundError(
            f"No matching files in {image_dir} and {mask_dir}. "
            "Confirm image/mask filenames and paths."
        )
    missing_masks = sorted(images.keys() - masks.keys())
    if missing_masks:
        print(f"Warning: {len(missing_masks)} image(s) have no matching mask.")
    return {k: images[k] for k in common}, {k: masks[k] for k in common}

class Nifti25DDataset(Dataset):
    """
    2.5D slice dataset: adjacent slices x three windows -> (9,H,W);
    target is the center slice mask -> (H,W).
    """
    def __init__(self, study_names, image_map, mask_map, cache_size=2,
                 negative_ratio=None, seed=42, num_classes=6):
        self.study_names = list(study_names)
        self.image_map, self.mask_map = image_map, mask_map
        self.cache_size = cache_size
        self.num_classes = int(num_classes)
        self.cache = OrderedDict()
        self.samples = []
        rng = np.random.default_rng(seed)

        for name in self.study_names:
            image = nib.load(str(self.image_map[name]))
            mask = nib.load(str(self.mask_map[name]))
            if len(image.shape) != 3 or len(mask.shape) != 3:
                raise ValueError(f"Expected 3D NIfTI pair: {name}")
            if image.shape != mask.shape:
                raise ValueError(f"Shape mismatch for {name}: {image.shape} vs {mask.shape}")
            depth = image.shape[2]
            if negative_ratio is None:
                chosen = np.arange(depth)
            else:
                m = np.asanyarray(mask.dataobj)
                positive = np.flatnonzero(np.any(m != 0, axis=(0, 1)))
                negative = np.flatnonzero(~np.any(m != 0, axis=(0, 1)))
                n_neg = min(len(negative), int(len(positive) * float(negative_ratio)))
                if n_neg:
                    negative = rng.choice(negative, size=n_neg, replace=False)
                else:
                    negative = np.array([], dtype=int)
                chosen = np.sort(np.concatenate([positive, negative]))
            self.samples.extend((name, int(z)) for z in chosen)

        if not self.samples:
            raise ValueError("Dataset has no slices. Check paths and sampling ratio.")
        print(f"Dataset: {len(self.study_names)} studies, {len(self.samples)} slices")

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        image_nii = nib.load(str(self.image_map[name]))
        mask_nii = nib.load(str(self.mask_map[name]))
        # Nibabel's dataobj applies NIfTI header scaling where present.
        image = np.asarray(image_nii.dataobj, dtype=np.float32)
        mask_raw = np.asanyarray(mask_nii.dataobj)
        mask = np.asarray(np.rint(mask_raw), dtype=np.int64)
        if image.shape != mask.shape:
            raise ValueError(f"Scaled image/mask shape mismatch for {name}")
        labels = np.unique(mask)
        if not np.isin(labels, np.arange(self.num_classes)).all():
            raise ValueError(f"Unexpected mask IDs in {name}: {labels.tolist()}")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite image values in {name}")
        item = (image, mask)
        self.cache[name] = item
        if self.cache_size is not None:
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return item

    def __getitem__(self, index):
        name, z = self.samples[index]
        image_vol, mask_vol = self._load_volume(name)
        depth = image_vol.shape[2]
        offsets = (-1, 0, 1)
        channels = []
        for offset in offsets:
            zi = min(max(z + offset, 0), depth - 1)
            channels.append(window_slice(image_vol[:, :, zi]))
        x = np.concatenate(channels, axis=0)  # (9,H,W)
        y = mask_vol[:, :, z]
        return torch.from_numpy(np.ascontiguousarray(x)).float(), \
               torch.from_numpy(np.ascontiguousarray(y)).long()

class NiftiPoolDataset(Nifti25DDataset):
    """AL scoring dataset: set negative_ratio=None to score every pool slice."""
    def __init__(self, study_names, image_map, mask_map, **kwargs):
        kwargs.pop("negative_ratio", None)
        super().__init__(study_names, image_map, mask_map,
                         negative_ratio=None, **kwargs)


class ImageOnlyNiftiPoolDataset(Dataset):
    """
    Strictly label-blind AL pool dataset.
    Reads image volumes only; it does not accept or open a mask directory.
    """
    def __init__(self, image_map, cache_size=1):
        self.image_map = image_map
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.samples = []
        for name, path in sorted(image_map.items()):
            nii = nib.load(str(path))
            if len(nii.shape) != 3:
                raise ValueError(f"Expected 3D NIfTI image: {name}")
            self.samples.extend((name, z) for z in range(nii.shape[2]))
        if not self.samples:
            raise ValueError("AL pool has no slices")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        volume = np.asarray(nib.load(str(self.image_map[name])).dataobj, dtype=np.float32)
        if not np.isfinite(volume).all():
            raise ValueError(f"Non-finite image values in AL study {name}")
        self.cache[name] = volume
        if self.cache_size is not None:
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return volume

    def __getitem__(self, index):
        name, z = self.samples[index]
        volume = self._load_image(name)
        depth = volume.shape[2]
        channels = [
            window_slice(volume[:, :, min(max(z + offset, 0), depth - 1)])
            for offset in (-1, 0, 1)
        ]
        x = np.concatenate(channels, axis=0)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), name, int(z)
