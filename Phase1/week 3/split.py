"""
============================================================
WEEK 3 — TRAIN / VALIDATION / ACTIVE LEARNING SPLIT
============================================================

Creates a volume-level split.

IMPORTANT:
Splitting is performed at VOLUME/PATIENT level,
NOT slice level.

This prevents slices from the same CT volume appearing
in both train and validation.

Outputs:

train_manifest.csv
val_manifest.csv
al_manifest.csv

and an explicit zero-overlap verification.
"""

from pathlib import Path
import random

import pandas as pd
import nibabel as nib


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


# ------------------------------------------------------------
# SPLIT RATIOS
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
AL_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# FIND PAIRS
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

    samples = []

    for image_path in image_files:

        mask_path = (
            MASK_DIR /
            image_path.name
        )

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

        if not mask_path.exists():
            continue

        samples.append(
            {
                "id": image_path.name,
                "image_path": str(
                    image_path
                ),
                "mask_path": str(
                    mask_path
                ),
            }
        )

    return pd.DataFrame(
        samples
    )


# ============================================================
# SPLIT
# ============================================================

def create_split(df):

    random.seed(
        RANDOM_SEED
    )

    ids = list(
        df["id"]
    )

    random.shuffle(
        ids
    )

    n = len(ids)

    n_train = int(
        n * TRAIN_RATIO
    )

    n_val = int(
        n * VAL_RATIO
    )

    train_ids = ids[
        :n_train
    ]

    val_ids = ids[
        n_train:
        n_train + n_val
    ]

    al_ids = ids[
        n_train + n_val:
    ]

    train = df[
        df["id"].isin(
            train_ids
        )
    ].copy()

    val = df[
        df["id"].isin(
            val_ids
        )
    ].copy()

    al = df[
        df["id"].isin(
            al_ids
        )
    ].copy()

    return train, val, al


# ============================================================
# ZERO OVERLAP CHECK
# ============================================================

def verify_zero_overlap(
    train,
    val,
    al
):

    train_ids = set(
        train["id"]
    )

    val_ids = set(
        val["id"]
    )

    al_ids = set(
        al["id"]
    )

    train_val = (
        train_ids &
        val_ids
    )

    train_al = (
        train_ids &
        al_ids
    )

    val_al = (
        val_ids &
        al_ids
    )

    print("\n")
    print("=" * 70)
    print("ZERO OVERLAP VERIFICATION")
    print("=" * 70)

    print(
        "Train ∩ Validation:",
        len(train_val)
    )

    print(
        "Train ∩ AL:",
        len(train_al)
    )

    print(
        "Validation ∩ AL:",
        len(val_al)
    )

    success = (
        len(train_val) == 0
        and
        len(train_al) == 0
        and
        len(val_al) == 0
    )

    output = (
        OUTPUT_DIR /
        "overlap_check.txt"
    )

    with open(
        output,
        "w",encoding="utf-8"
    ) as f:

        f.write(
            "ZERO OVERLAP CHECK\n"
        )

        f.write(
            "==================\n\n"
        )

        f.write(
            f"Train ∩ Validation: "
            f"{len(train_val)}\n"
        )

        f.write(
            f"Train ∩ AL: "
            f"{len(train_al)}\n"
        )

        f.write(
            f"Validation ∩ AL: "
            f"{len(val_al)}\n"
        )

        f.write(
            f"\nZERO OVERLAP: "
            f"{success}\n"
        )

    if success:

        print(
            "\n✓ ZERO OVERLAP GUARANTEE PASSED"
        )

    else:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED!"
        )

    print(
        f"Saved: {output}"
    )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    train,
    val,
    al
):

    train.to_csv(
        OUTPUT_DIR /
        "train_manifest.csv",
        index=False
    )

    val.to_csv(
        OUTPUT_DIR /
        "val_manifest.csv",
        index=False
    )

    al.to_csv(
        OUTPUT_DIR /
        "al_manifest.csv",
        index=False
    )

    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "volume_count": len(train),
            },
            {
                "split": "validation",
                "volume_count": len(val),
            },
            {
                "split": "active_learning",
                "volume_count": len(al),
            },
        ]
    )

    summary.to_csv(
        OUTPUT_DIR /
        "split_summary.csv",
        index=False
    )

    print("\n")
    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = find_pairs()

    print(
        f"Total volumes: {len(df)}"
    )

    if len(df) == 0:

        raise RuntimeError(
            "No volumes found."
        )

    train, val, al = create_split(
        df
    )

    save_splits(
        train,
        val,
        al
    )

    verify_zero_overlap(
        train,
        val,
        al
    )

    print(
        "\n✓ Split generation complete."
    )


if __name__ == "__main__":
    main()