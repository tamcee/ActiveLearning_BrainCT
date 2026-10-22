"""
============================================================
WEEK 3 — EDA + DATA STRATEGY FINALISATION
============================================================

BHSD NIfTI Dataset

Computes:

1. Per-class volume count
2. Per-class slice count
3. Per-class pixel count
4. Pixel fraction of total
5. Volume-level imbalance
6. Pixel-level imbalance
7. Classes with <50 training volumes
8. Dataset-level statistics

Expected structure:

label_192/
├── images/
│   ├── *.nii.gz
│   └── ...
│
└── ground truths/
    ├── *.nii.gz
    └── ...

Output:

outputs/week3/
├── class_statistics.csv
├── volume_imbalance.png
├── pixel_imbalance.png
└── low_volume_classes.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path(
    "Dataset/labeled/images"
)

MASK_DIR = Path(
    "Dataset/labeled/ground_truths"
)

OUTPUT_DIR = Path(
    "outputs/week3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# NIFTI LOADING
# ============================================================

def load_nifti(path):

    nii = nib.load(str(path))

    data = nii.get_fdata(
        dtype=np.float32
    )

    return data, nii


# ============================================================
# FIND IMAGE / MASK PAIRS
# ============================================================

def find_pairs():

    image_files = sorted(
        list(
            IMAGE_DIR.glob("*.nii.gz")
        )
        +
        list(
            IMAGE_DIR.glob("*.nii")
        )
    )

    pairs = []

    for image_path in image_files:

        mask_path = (
            MASK_DIR /
            image_path.name
        )

        # Handle extension mismatch
        if not mask_path.exists():

            if image_path.name.endswith(
                ".nii.gz"
            ):

                alternative = (
                    MASK_DIR /
                    image_path.name.replace(
                        ".nii.gz",
                        ".nii"
                    )
                )

            else:

                alternative = (
                    MASK_DIR /
                    image_path.name.replace(
                        ".nii",
                        ".nii.gz"
                    )
                )

            if alternative.exists():

                mask_path = alternative

        if mask_path.exists():

            pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

    return pairs


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(pairs):

    print("\n")
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Image-mask pairs : {len(pairs)}"
    )

    if not pairs:
        raise RuntimeError(
            "No image-mask pairs found."
        )

    volume_shapes = []
    spacings = []

    for image_path, mask_path in pairs:

        image, image_nii = load_nifti(
            image_path
        )

        mask, mask_nii = load_nifti(
            mask_path
        )

        if image.shape != mask.shape:

            print(
                "WARNING shape mismatch:",
                image_path.name
            )

        volume_shapes.append(
            image.shape
        )

        spacings.append(
            image_nii.header.get_zooms()[:3]
        )

    unique_shapes = {
        tuple(x)
        for x in volume_shapes
    }

    unique_spacings = {
        tuple(
            round(float(v), 4)
            for v in x
        )
        for x in spacings
    }

    print(
        "Unique volume shapes:",
        unique_shapes
    )

    print(
        "Unique voxel spacings:",
        unique_spacings
    )

    print("=" * 70)


# ============================================================
# CLASS STATISTICS
# ============================================================

def compute_class_statistics(pairs):

    """
    Computes:

    volume_count
        Number of volumes containing the class.

    slice_count
        Number of slices containing the class.

    pixel_count
        Total number of pixels belonging to class.

    pixel_fraction
        Class pixels / all foreground pixels.

    """

    # First discover classes
    all_classes = set()

    for _, mask_path in pairs:

        mask, _ = load_nifti(
            mask_path
        )

        unique = np.unique(mask)

        for value in unique:

            value = int(round(value))

            if value != 0:
                all_classes.add(value)

    classes = sorted(
        all_classes
    )

    print(
        "\nDetected classes:",
        classes
    )

    statistics = []

    total_foreground_pixels = 0

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    for class_id in classes:

        volume_count = 0
        slice_count = 0
        pixel_count = 0

        for image_path, mask_path in pairs:

            mask, _ = load_nifti(
                mask_path
            )

            class_mask = (
                mask == class_id
            )

            # Volume level
            if np.any(class_mask):

                volume_count += 1

            # Pixel level
            pixels = int(
                np.sum(class_mask)
            )

            pixel_count += pixels

            # Slice level
            slices = np.any(
                class_mask,
                axis=(0, 1)
            )

            slice_count += int(
                np.sum(slices)
            )

        total_foreground_pixels += (
            pixel_count
        )

        statistics.append(
            {
                "class": class_id,
                "volume_count": volume_count,
                "slice_count": slice_count,
                "pixel_count": pixel_count,
            }
        )

    # --------------------------------------------------------
    # Pixel fractions
    # --------------------------------------------------------

    for row in statistics:

        row["pixel_fraction"] = (
            row["pixel_count"]
            /
            total_foreground_pixels
            if total_foreground_pixels > 0
            else 0
        )

    df = pd.DataFrame(
        statistics
    )

    return df


# ============================================================
# SAVE CLASS TABLE
# ============================================================

def save_statistics(df):

    output = (
        OUTPUT_DIR /
        "class_statistics.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved: {output}"
    )

    print("\n")
    print(df.to_string(index=False))


# ============================================================
# VOLUME LEVEL IMBALANCE
# ============================================================

def plot_volume_imbalance(df):

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        df["class"].astype(str),
        df["volume_count"]
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of volumes"
    )

    plt.title(
        "BHSD Class Imbalance — Volume Level"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR /
        "volume_imbalance.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# PIXEL LEVEL IMBALANCE
# ============================================================

def plot_pixel_imbalance(df):

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        df["class"].astype(str),
        df["pixel_fraction"]
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Fraction of foreground pixels"
    )

    plt.title(
        "BHSD Class Imbalance — Pixel Level"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR /
        "pixel_imbalance.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# LOW VOLUME CLASSES
# ============================================================

def identify_low_volume_classes(
    df,
    threshold=50
):

    low_volume = df[
        df["volume_count"] < threshold
    ].copy()

    low_volume[
        "high_risk_low_dice"
    ] = True

    output = (
        OUTPUT_DIR /
        "low_volume_classes.csv"
    )

    low_volume.to_csv(
        output,
        index=False
    )

    print("\n")
    print("=" * 70)
    print(
        f"CLASSES WITH < {threshold} TRAINING VOLUMES"
    )
    print("=" * 70)

    if len(low_volume) == 0:

        print(
            "No classes below threshold."
        )

    else:

        print(
            low_volume.to_string(
                index=False
            )
        )

    print(
        f"\nSaved: {output}"
    )


# ============================================================
# OVERALL DATASET STATISTICS
# ============================================================

def calculate_dataset_statistics(
    pairs
):

    total_slices = 0
    total_pixels = 0
    total_foreground = 0

    for image_path, mask_path in pairs:

        image, _ = load_nifti(
            image_path
        )

        mask, _ = load_nifti(
            mask_path
        )

        total_slices += image.shape[2]

        total_pixels += np.prod(
            mask.shape
        )

        total_foreground += np.sum(
            mask != 0
        )

    return {
        "volumes": len(pairs),
        "slices": total_slices,
        "pixels": int(total_pixels),
        "foreground_pixels": int(
            total_foreground
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WEEK 3 — BHSD EDA"
    )

    print(
        "=" * 70
    )

    pairs = find_pairs()

    if not pairs:

        raise RuntimeError(
            "No image-mask pairs found."
        )

    dataset_summary(
        pairs
    )

    stats = calculate_dataset_statistics(
        pairs
    )

    print("\nDATASET TOTALS")

    for key, value in stats.items():

        print(
            f"{key}: {value}"
        )

    df = compute_class_statistics(
        pairs
    )

    save_statistics(
        df
    )

    plot_volume_imbalance(
        df
    )

    plot_pixel_imbalance(
        df
    )

    identify_low_volume_classes(
        df,
        threshold=50
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WEEK 3 EDA COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()