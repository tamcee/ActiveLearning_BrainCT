# Brain CT Hemorrhage Detection and Segmentation

Research code and experiment artifacts for detecting and segmenting intracranial hemorrhage in head CT volumes. The repository documents a staged workflow that starts with CT preprocessing and dataset analysis, then evaluates baseline semantic-segmentation pipelines using a 2.5D U-Net, DeepLabV3, and nnU-Net.

> **Research use only.** This repository is not a medical device and must not be used for diagnosis, treatment, triage, or other clinical decision-making. The included baselines, labels, splits, and metrics require further validation before any real-world use.

## Project status

The project currently contains:

- Multi-window CT preprocessing with brain, subdural, and bone windows.
- Unit tests for Hounsfield Unit (HU) windowing and ImageNet-style normalization.
- Exploratory data analysis at volume, slice, and pixel levels.
- Volume-level train, validation, and active-learning splits with zero-overlap verification.
- Baseline experiments for:
  - A 2.5D U-Net.
  - DeepLabV3.
  - nnU-Net.
- Training histories, validation metrics, confusion matrices, prediction overlays, and experiment reports.

The repository contains development and smoke-test results. Some experiment metadata explicitly identifies unresolved dataset provenance, label semantics, intensity conventions, and active-learning overlap checks. Treat all reported results as provisional.

## Objectives

The project is designed to support the following research questions:

1. Can multi-window CT representations improve hemorrhage segmentation?
2. How do conventional segmentation baselines compare with nnU-Net?
3. How does class imbalance affect volume-level, slice-level, and pixel-level performance?
4. Can volume-level data partitioning provide a reliable foundation for active-learning experiments?

## Dataset layout

The expected local dataset layout is:

```text
Dataset/
└── labeled/
    ├── images/
    │   ├── <study_id>.nii
    │   └── <study_id>.nii.gz
    └── ground_truths/
        ├── <study_id>.nii
        └── <study_id>.nii.gz
```

Each image volume should have a corresponding ground-truth volume with the same study filename. The split-generation script also supports matching `.nii` and `.nii.gz` alternatives.

The medical imaging dataset is not intended to be redistributed through this repository. Obtain and use the data only in accordance with its source license, institutional approvals, privacy requirements, and applicable data-protection policies.

## Repository structure

```text
.
├── Dataset/                         # Local imaging data; ignored by Git
├── data_splits/                     # Development manifests
├── Phase1/
│   ├── week 2/
│   │   ├── preprocessing.py         # HU loading, windowing, normalization, dataset helpers
│   │   ├── test_preprocessing.py    # Preprocessing unit tests
│   │   └── visualize_preprocessing.ipynb
│   └── week 3/
│       ├── eda.py                   # Class-imbalance and dataset analysis
│       ├── split.py                 # Train/validation/active-learning split generation
│       ├── report.py                # Markdown report generation
│       └── outputs/                 # Manifests, tables, plots, and reports
├── Phase2_baselines/
│   ├── unet_25d_training.ipynb      # 2.5D U-Net baseline
│   ├── deeplabv3_pipeline.ipynb     # DeepLabV3 baseline
│   ├── nnunet_complete.ipynb        # nnU-Net preparation/training workflow
│   ├── nnunet_week4_pipeline.ipynb
│   ├── Unet_week4_outputs/          # U-Net configuration, metrics, and TensorBoard logs
│   ├── DeepLabV3_outputs/            # DeepLabV3 metrics and figures
│   └── nnunet_output_week4/          # nnU-Net logs, reports, and prediction overlays
├── Deliverables/                    # Project reports and presentations
├── .gitignore                       # Rules for datasets, checkpoints, caches, and local files
└── README.md
```

Large datasets, generated checkpoints, cache directories, and training logs are intentionally excluded from version control. See [`.gitignore`](./.gitignore) for the complete list.

## Preprocessing

The preprocessing module in [`Phase1/week 2/preprocessing.py`](./Phase1/week%202/preprocessing.py) provides:

- CT volume loading and HU conversion.
- Three-window intensity representation:
  - **Brain:** window level 40, width 80.
  - **Subdural:** window level 75, width 215.
  - **Bone:** window level 400, width 1800.
- A three-channel window stack with values clipped to `[0, 1]`.
- Optional ImageNet mean and standard-deviation normalization.
- Dataset and dataloader helpers for segmentation experiments.

For the 2.5D U-Net baseline, each input sample is configured as nine channels: three window channels for each of three neighboring slices. The documented baseline uses a `512 x 512` spatial resolution and produces six output classes, including background.

## Data analysis and splitting

The Week 3 workflow measures imbalance at three separate levels:

- **Volume level:** independent study/volume counts.
- **Slice level:** number of slices containing each class.
- **Pixel level:** foreground pixel counts and fractions.

The documented data strategy produced the following provisional split:

| Split | Volumes |
| --- | ---: |
| Training | 134 |
| Validation | 28 |
| Active learning | 30 |

Splits are generated at volume level rather than slice level. This prevents slices from the same CT volume from appearing in multiple partitions and reduces the risk of data leakage. The script writes an explicit intersection check and fails if any split overlaps another.

Run the analysis workflow from the repository root:

```powershell
python "Phase1/week 3/eda.py"
python "Phase1/week 3/split.py"
python "Phase1/week 3/report.py"
```

The scripts currently use relative paths and expect the `Dataset/` directory to exist in the working directory. Generated files include:

```text
outputs/week3/
├── train_manifest.csv
├── val_manifest.csv
├── al_manifest.csv
├── split_summary.csv
├── overlap_check.txt
├── class_statistics.csv
├── low_volume_classes.csv
└── week3_final_report.md
```

The repository also contains exported manifests under [`data_splits/`](./data_splits/) and [`Phase1/week 3/outputs/`](./Phase1/week%203/outputs/). Confirm which split is appropriate before starting a new experiment.

## Baseline experiments

### 2.5D U-Net

The U-Net development baseline uses:

- Nine input channels from three neighboring slices and three HU windows.
- Six output classes.
- Cross-entropy plus foreground multiclass soft Dice loss.
- AdamW optimization.
- Cosine annealing learning-rate scheduling.
- Mixed precision and optional compilation.
- Seed `42`.

The configuration and model card are stored in [`Phase2_baselines/Unet_week4_outputs/`](./Phase2_baselines/Unet_week4_outputs/). The model card reports a provisional best validation macro-Dice score of approximately `0.426`; it also states that the run is a smoke test and not a final performance evaluation.

### DeepLabV3

DeepLabV3 evaluation artifacts are available under [`Phase2_baselines/DeepLabV3_outputs/`](./Phase2_baselines/DeepLabV3_outputs/), including:

- Training history.
- Per-class validation metrics.
- Confusion matrices.
- Training curves.
- Dataset inspection images.
- Prediction overlays.

The tracked validation table reports per-class Dice, IoU, precision, and recall for background and five foreground class IDs. Class `3` is reported with zero Dice and IoU in that table, highlighting the need for class-specific analysis rather than relying on aggregate scores alone.

### nnU-Net

The nnU-Net workflow and outputs are under [`Phase2_baselines/`](./Phase2_baselines/) and [`Phase2_baselines/nnunet_output_week4/`](./Phase2_baselines/nnunet_output_week4/). The recorded training log describes:

- A 2D configuration generated by nnU-Net.
- Nine input channels.
- A `512 x 448` patch size.
- Fold `0` from a five-fold split.
- 129 training cases and 33 validation cases for that run.

When using or extending the nnU-Net workflow, cite:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18(2), 203–211.

## Installation

No lockfile or requirements manifest is currently included, so the exact environment must be recorded when reproducing an experiment. A typical environment should provide:

- Python 3.10 or newer.
- NumPy.
- pandas.
- pytest.
- nibabel.
- PyTorch and its platform-appropriate dependencies.
- Jupyter Notebook or JupyterLab for notebook workflows.
- nnU-Net and its dependencies for the nnU-Net pipeline.

Create an isolated environment, then install the dependencies appropriate for your operating system and accelerator:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas pytest nibabel jupyter
```

Install PyTorch using the command recommended by the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Install nnU-Net separately if you plan to run that pipeline, following the [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet).

The recorded U-Net metadata was produced with Python `3.14.7` and PyTorch `2.11.0+cu128`; these versions describe one development environment and are not a guarantee of compatibility on another machine.

## Verification

Run the preprocessing tests from their source directory:

```powershell
Set-Location "Phase1\week 2"
python -m pytest test_preprocessing.py -v
Set-Location ..\..
```

The tests cover:

- Window-level boundary calculations.
- Known HU-to-window conversions.
- Output shape and range guarantees.
- ImageNet normalization.
- Invalid channel and dimensionality handling.
- End-to-end windowing followed by normalization.

Before trusting a new experiment, also verify:

1. Image and mask pairs are complete.
2. Train, validation, and active-learning manifests have no volume overlap.
3. Label IDs and their clinical meanings are confirmed.
4. Preprocessing and padding conventions match the selected model.
5. Metrics are reported per class as well as in aggregate.
6. The exact code revision, environment, seed, split, and configuration are recorded.

## Reproducing notebook experiments

The main notebook entry points are:

| Notebook | Purpose |
| --- | --- |
| [`visualize_preprocessing.ipynb`](./Phase1/week%202/visualize_preprocessing.ipynb) | Inspect CT preprocessing and windowed representations |
| [`unet_25d_training.ipynb`](./Phase2_baselines/unet_25d_training.ipynb) | Train/evaluate the 2.5D U-Net baseline |
| [`deeplabv3_pipeline.ipynb`](./Phase2_baselines/deeplabv3_pipeline.ipynb) | Run the DeepLabV3 baseline |
| [`nnunet_complete.ipynb`](./Phase2_baselines/nnunet_complete.ipynb) | Prepare and run the nnU-Net workflow |
| [`nnunet_week4_pipeline.ipynb`](./Phase2_baselines/nnunet_output_week4/nnunet_week4_pipeline.ipynb) | Week 4 nnU-Net pipeline and analysis |

Notebook outputs may depend on local paths, GPU availability, installed package versions, and data layout. Review and update path variables before running a notebook on a new machine.

## Evaluation and reporting

The repository includes the following evaluation artifacts:

- Per-class Dice and IoU.
- Precision and recall.
- Validation confusion matrices.
- Training and validation loss histories.
- Prediction overlays.
- TensorBoard event files.
- Dataset inspection figures.
- Data-split and class-imbalance reports.

For medical image segmentation, report background and foreground classes separately. A high background score can conceal poor performance on rare hemorrhage classes, especially when class imbalance is substantial.

## Limitations

The current artifacts identify several limitations:

- Some class-to-clinical-label mappings remain unconfirmed.
- Dataset provenance and intensity conventions require confirmation.
- Several runs are smoke tests or early development experiments.
- The data are imbalanced, including a low-volume class.
- Active-learning pool overlap requires explicit verification before use.
- Results may depend on local GPU, CUDA, PyTorch, and nnU-Net versions.
- Checkpoints are ignored and are not distributed through GitHub.
- Aggregate metrics alone are insufficient to assess clinical usefulness.

## Data and artifact policy

The following generated or local-only content is ignored by Git:

- `Dataset/`
- Model checkpoints such as `*.pth`, `*.pt`, and `*.ckpt`.
- TensorBoard logs and training logs.
- Python caches and compiled files.
- Virtual environments.
- Notebook checkpoints.
- IDE, operating-system, temporary, and environment files.

Do not force-add patient data, secrets, credentials, or large model artifacts to the repository. Use an approved artifact store or Git LFS only after confirming the project’s data-governance and storage requirements.

## Contributing

When adding a new experiment:

1. Keep source code, configuration, and result summaries separate.
2. Record the dataset revision, split manifest, random seed, software versions, and hardware.
3. Add tests for preprocessing or data-integrity changes.
4. Verify volume-level split isolation.
5. Report per-class metrics and document known failure modes.
6. Keep datasets, secrets, checkpoints, and transient logs out of Git.

## License and responsible use

No project license is currently included. Before redistributing code, reports, data, or derived artifacts, confirm the applicable licenses and permissions for every component and dataset. The project should be used only for responsible research and development under appropriate institutional oversight.
