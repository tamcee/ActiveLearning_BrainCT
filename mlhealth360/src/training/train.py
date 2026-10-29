"""Training entry point for the existing 2.5D NIfTI pipeline."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import yaml
from src.data.dataset import pair_nifti_files, Nifti25DDataset
from src.models.unet import UNet25D, count_dropout
from src.training.loss import combined_loss
from src.training.checkpoint import save_checkpoint
from src.evaluation.metrics import update_confusion_matrix, metrics_from_confusion

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def validate(model, loader, device, num_classes, dice_weight):
    model.eval()
    total_loss, batches = 0.0, 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss, _ = combined_loss(logits, y, dice_weight, num_classes)
            total_loss += loss.item(); batches += 1
            update_confusion_matrix(cm, logits.argmax(1), y, num_classes)
    if not batches: raise RuntimeError("Validation loader is empty")
    metrics = metrics_from_confusion(cm.cpu().numpy())
    metrics["loss"] = total_loss / batches
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    args = parser.parse_args()
    ds_cfg = yaml.safe_load(open(args.dataset_config, encoding="utf-8"))
    model_cfg = yaml.safe_load(open(args.model_config, encoding="utf-8"))["model"]
    tr_cfg = yaml.safe_load(open(args.training_config, encoding="utf-8"))["training"]
    seed_everything(tr_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_map, mask_map = pair_nifti_files(ds_cfg["dataset"]["image_dir"], ds_cfg["dataset"]["mask_dir"])
    names = sorted(image_map)
    train_names, val_names = train_test_split(names, test_size=ds_cfg["dataset"]["validation_fraction"],
                                               random_state=ds_cfg["dataset"]["seed"])
    # Save exact study split for reproducibility.
    out_dir = Path(tr_cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "development_split.json").write_text(
        json.dumps({"train_studies": sorted(train_names), "val_studies": sorted(val_names)}, indent=2),
        encoding="utf-8")
    train_ds = Nifti25DDataset(train_names, image_map, mask_map,
        cache_size=ds_cfg["dataset"]["cache_size"],
        negative_ratio=tr_cfg.get("train_negative_ratio", None), seed=tr_cfg["seed"])
    val_ds = Nifti25DDataset(val_names, image_map, mask_map,
        cache_size=ds_cfg["dataset"]["cache_size"], negative_ratio=None, seed=tr_cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=tr_cfg["batch_size"], shuffle=True,
        num_workers=tr_cfg["num_workers"], pin_memory=tr_cfg["pin_memory"])
    val_loader = DataLoader(val_ds, batch_size=tr_cfg["batch_size"], shuffle=False,
        num_workers=tr_cfg["num_workers"], pin_memory=tr_cfg["pin_memory"])

    model = UNet25D(**{k: model_cfg[k] for k in ("in_channels","num_classes","base_channels","dropout_p")}).to(device)
    if count_dropout(model) < 1: raise RuntimeError("MC Dropout requires at least one dropout layer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg["learning_rate"], weight_decay=tr_cfg["weight_decay"])
    epochs = int(tr_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if tr_cfg["scheduler"] == "cosine" else None
    amp = bool(tr_cfg["amp"] and device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    ckpt_dir = Path(tr_cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0; history = []

    for epoch in range(1, epochs+1):
        model.train(); loss_sum = 0.; batches = 0; started = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(x)
                loss, _ = combined_loss(logits, y, tr_cfg["dice_weight"], model_cfg["num_classes"])
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item(); batches += 1
        if not batches: raise RuntimeError("Training loader is empty")
        if scheduler: scheduler.step()
        val = validate(model, val_loader, device, model_cfg["num_classes"], tr_cfg["dice_weight"])
        record = {"epoch": epoch, "train_loss": loss_sum/batches, "val_loss": val["loss"],
                  "val_macro_dice": val["macro_foreground_dice"], "per_class": val["per_class"],
                  "seconds": time.time()-started, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        save_checkpoint(ckpt_dir/"last_model.pt", model, optimizer, scheduler, epoch, best, tr_cfg, record, scaler)
        if np.isfinite(record["val_macro_dice"]) and record["val_macro_dice"] > best:
            best = record["val_macro_dice"]
            save_checkpoint(ckpt_dir/"best_model.pt", model, optimizer, scheduler, epoch, best, tr_cfg, record, scaler)
        print(f"Epoch {epoch:03d}/{epochs} train={record['train_loss']:.4f} val={record['val_loss']:.4f} macroDice={record['val_macro_dice']:.4f} {record['seconds']:.1f}s")
        (out_dir/"history.json").write_text(json.dumps(history, indent=2, allow_nan=True), encoding="utf-8")
    print("Best checkpoint:", ckpt_dir/"best_model.pt")

if __name__ == "__main__":
    main()
