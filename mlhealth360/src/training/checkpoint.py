"""Checkpoint helpers."""
from pathlib import Path
import torch

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0,
                    best_macro_dice=float("nan"), config=None, metrics=None,
                    scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "best_macro_dice": float(best_macro_dice),
        "config": config or {},
        "metrics": metrics or {},
    }
    if optimizer is not None: payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None: payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None: payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)
    return path

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
