# mLHealth 360 — 2.5D U-Net + Active Learning scaffold

This repository extracts the working choices from `unet_25d_training.ipynb`
into importable modules and CLI scripts.

## Important assumptions
- Current dataset is paired 3D NIfTI (`.nii.gz`) image and mask volumes.
- Image and mask filenames match exactly.
- Image and mask array axes are aligned and mask IDs are integers `0..5`.
- Image values are already HU after NIfTI scaling, as in the existing notebook's
  `load_volume_as_hu` path. Verify this with dataset documentation before training.
- Class IDs `1..5` are currently called `Label_1..Label_5`. Confirm the clinical
  mapping with the project team before naming them IPH/SAH/EDH/SDH/IVH.
- This is a research project scaffold, not a medical device or for clinical use.

## Layout
- `src/data`: HU conversion, DICOM helpers, windowing, 2.5D NIfTI dataset.
- `src/models`: the notebook's 2.5D U-Net.
- `src/training`: CE + Dice loss, checkpointing, training entry point.
- `src/evaluation`: metrics and overlay visualization.
- `src/active_learning`: MC Dropout, entropy/BALD/variance, filtering and selection.
- `run_al.py`: score-table filtering and selection manifest.
- `predict.py`: inference on a NIfTI volume.

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Arrange data
```text
data/
  train/
    images/      # paired study.nii.gz
    masks/       # same filenames
  al_pool/
    images/      # different dataset; never used for training before selection
    masks/       # held back during scoring; used only for simulated annotation
```
Do not put the AL pool in `data/train`. Check dataset licence, provenance,
clinical label mapping, geometry, and zero study/patient overlap before running.

## Train
Adjust paths in `configs/dataset.yaml` and run from repository root:
```bash
python -m src.training.train
```
This creates a reproducible study-level split and writes checkpoints under
`experiments/baseline/checkpoints/`.

## Predict
```bash
python predict.py --image data/train/images/EXAMPLE.nii.gz \
  --checkpoint experiments/baseline/checkpoints/best_model.pt \
  --output experiments/baseline/prediction.nii.gz
```

## Active Learning
Score an entirely separate pool:
```bash
python -m src.active_learning.score_pool \
  --checkpoint experiments/baseline/checkpoints/best_model.pt \
  --image-dir data/al_pool/images \
  --output experiments/active_learning/scores.csv
python run_al.py --scores experiments/active_learning/scores.csv \
  --output experiments/active_learning/selected_for_training.csv --budget 150
```
Scoring uses `ImageOnlyNiftiPoolDataset` and never opens mask files. Keep the
pool's mask directory out of the scoring command. The selected CSV is a manifest,
not a set of revealed labels; reveal labels only after selection in the simulated
annotation stage.

## Tests
```bash
pytest -q
```

## Notebooks
The `notebooks/` files are learning/exploration templates. Run the scripts for
the main training and AL pipeline; notebooks should not become a second,
divergent implementation.

## Known next engineering steps
1. Add a manifest-driven dataset config instead of relying only on directory globbing.
2. Verify NIfTI affine/orientation and resampling policy for image-mask pairs.
3. Make AL scoring image-only so hidden labels are not loaded in memory.
4. Add a proper selected-slice export/retraining Dataset; the included retrain CLI
   expects selected cases to be assembled as full paired NIfTI volumes.
5. Add random-baseline trials across at least three seeds and generate the
   label-efficiency curves before making label-saving claims.
