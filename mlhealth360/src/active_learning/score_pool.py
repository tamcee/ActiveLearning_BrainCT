"""Score an AL pool without opening or reading any ground-truth masks."""
import argparse
from pathlib import Path
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from src.data.dataset import ImageOnlyNiftiPoolDataset
from src.models.unet import UNet25D
from src.training.checkpoint import load_checkpoint
from src.active_learning.uncertainty import score_batch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--output", default="experiments/active_learning/scores.csv")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--passes", type=int, default=10)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.model_config, encoding="utf-8"))["model"]
    image_dir = Path(args.image_dir)
    image_map = {p.name: p for p in image_dir.glob("*.nii.gz")}
    if not image_map:
        raise FileNotFoundError(f"No .nii.gz images found in {image_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet25D(**{k: cfg[k] for k in ("in_channels", "num_classes", "base_channels", "dropout_p")}).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    ds = ImageOnlyNiftiPoolDataset(image_map, cache_size=1)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    for batch_idx, (x, studies, slices) in enumerate(loader):
        x = x.to(device)
        result = score_batch(model, x, passes=args.passes, foreground_only=True)
        pred = result["prediction"]
        for j in range(x.shape[0]):
            rows.append({
                "study": studies[j],
                "slice": int(slices[j]),
                "entropy": float(result["scores"]["entropy"][j].cpu()),
                "bald": float(result["scores"]["bald"][j].cpu()),
                "variance": float(result["scores"]["variance"][j].cpu()),
                "predicted_foreground_pixels": int((pred[j] > 0).sum().cpu()),
            })
        if (batch_idx + 1) % 20 == 0:
            print(f"Scored {len(rows)}/{len(ds)} slices")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}; rows={len(rows)}")
    print("Label-blind scoring complete: only image files were read.")

if __name__ == "__main__":
    main()
