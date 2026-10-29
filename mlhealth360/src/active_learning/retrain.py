"""Fine-tune from a baseline checkpoint after selected AL labels are revealed."""
import argparse, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from src.data.dataset import pair_nifti_files, Nifti25DDataset
from src.models.unet import UNet25D
from src.training.loss import combined_loss
from src.training.checkpoint import load_checkpoint, save_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-config", default="configs/dataset.yaml")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--baseline", required=True)
    p.add_argument("--selected-image-dir", required=True)
    p.add_argument("--selected-mask-dir", required=True)
    p.add_argument("--buffer-image-dir", required=True)
    p.add_argument("--buffer-mask-dir", required=True)
    p.add_argument("--output", default="experiments/active_learning/model_v2.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.model_config, encoding="utf-8"))["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet25D(**{k:cfg[k] for k in ("in_channels","num_classes","base_channels","dropout_p")}).to(device)
    load_checkpoint(args.baseline, model, map_location=device)
    selected_i, selected_m = pair_nifti_files(args.selected_image_dir, args.selected_mask_dir)
    buffer_i, buffer_m = pair_nifti_files(args.buffer_image_dir, args.buffer_mask_dir)
    # This script expects each selected item as a full NIfTI volume. For slice-only
    # exports, first assemble consistent 3D volumes or implement a slice dataset.
    selected_ds = Nifti25DDataset(sorted(selected_i), selected_i, selected_m, negative_ratio=None)
    buffer_ds = Nifti25DDataset(sorted(buffer_i), buffer_i, buffer_m, negative_ratio=None)
    loader = DataLoader(ConcatDataset([selected_ds, buffer_ds]), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            loss,_=combined_loss(model(x),y,num_classes=cfg["num_classes"])
            loss.backward(); opt.step()
        print(f"Fine-tune epoch {epoch+1}/{args.epochs}")
    save_checkpoint(args.output, model, opt, epoch=args.epochs, config=cfg)
    print("Saved", args.output)

if __name__ == "__main__": main()
