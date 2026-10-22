"""
Generate a Markdown Week 3 data strategy report
from the outputs produced by the EDA and split scripts.
"""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(
    "outputs/week3"
)


def main():

    stats_file = (
        OUTPUT_DIR /
        "class_statistics.csv"
    )

    split_file = (
        OUTPUT_DIR /
        "split_summary.csv"
    )

    low_file = (
        OUTPUT_DIR /
        "low_volume_classes.csv"
    )

    overlap_file = (
        OUTPUT_DIR /
        "overlap_check.txt"
    )

    if not stats_file.exists():

        raise RuntimeError(
            "Run week3_eda.py first."
        )

    stats = pd.read_csv(
        stats_file
    )

    split = pd.read_csv(
        split_file
    )

    low = pd.read_csv(
        low_file
    )

    overlap = ""

    if overlap_file.exists():

        overlap = overlap_file.read_text()

    report = []

    report.append(
        "# Week 3 — EDA + Data Strategy Finalisation\n"
    )

    report.append(
        "## 1. Class Imbalance\n"
    )

    report.append(
        "The following table reports imbalance at volume, "
        "slice and pixel levels.\n"
    )

    report.append(
        stats.to_markdown(
            index=False
        )
    )

    report.append(
        "\n## 2. Low-Volume Classes\n"
    )

    if len(low) > 0:

        report.append(
            "The following classes contain fewer than "
            "50 training volumes and should be considered "
            "high-risk for unstable segmentation performance.\n"
        )

        report.append(
            low.to_markdown(
                index=False
            )
        )

    else:

        report.append(
            "No classes contain fewer than 50 training volumes."
        )

    report.append(
        "\n## 3. Dataset Split\n"
    )

    report.append(
        split.to_markdown(
            index=False
        )
    )

    report.append(
        "\n## 4. Zero Overlap Verification\n"
    )

    report.append(
        "```text\n"
        +
        overlap
        +
        "\n```"
    )

    report.append(
        "\n## 5. Data Strategy\n"
    )

    report.append(
        """
The analysis distinguishes three levels of class imbalance:

1. Volume-level imbalance
2. Slice-level imbalance
3. Pixel-level imbalance

These measurements are treated separately because they affect
different modelling decisions.

Volume-level imbalance affects whether the model has sufficient
independent patient/volume examples.

Slice-level imbalance affects the number of training examples
containing each class.

Pixel-level imbalance reflects the spatial dominance of large
background or hemorrhage regions.

Training, validation and active-learning pools are split at
volume level to prevent slices from the same volume appearing
in different partitions.
"""
    )

    report.append(
        "\n## 6. Compatibility\n"
    )

    compatibility_file = (
        OUTPUT_DIR /
        "compatibility_report.csv"
    )

    if compatibility_file.exists():

        compatibility = pd.read_csv(
            compatibility_file
        )

        report.append(
            compatibility.to_markdown(
                index=False
            )
        )

    else:

        report.append(
            "Compatibility analysis has not yet been run."
        )

    output = (
        OUTPUT_DIR /
        "week3_final_report.md"
    )

    output.write_text(
        "\n\n".join(report)
    )

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()