"""Segmentation metrics from a global pixel confusion matrix."""
import numpy as np
import torch

def update_confusion_matrix(confusion, predictions, targets, num_classes=6):
    # Rows=true class, columns=predicted class.
    encoded = targets.reshape(-1).long() * num_classes + predictions.reshape(-1).long()
    confusion += torch.bincount(encoded, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    return confusion

def metrics_from_confusion(confusion, class_names=None):
    cm = np.asarray(confusion, dtype=np.float64)
    n = cm.shape[0]
    if class_names is None:
        class_names = ["Background"] + [f"Label_{i}" for i in range(1, n)]
    rows = []
    for c in range(n):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        dice_den = 2*tp + fp + fn
        precision = tp/(tp+fp) if tp+fp else float("nan")
        recall = tp/(tp+fn) if tp+fn else float("nan")
        dice = 2*tp/dice_den if dice_den else float("nan")
        rows.append({"class_id": c, "class_name": class_names[c],
                     "dice": dice, "precision": precision, "recall": recall,
                     "ground_truth_pixels": int(cm[c, :].sum()),
                     "predicted_pixels": int(cm[:, c].sum())})
    foreground = [r["dice"] for r in rows[1:] if np.isfinite(r["dice"])]
    return {"per_class": rows, "macro_foreground_dice": float(np.mean(foreground)) if foreground else float("nan"),
            "confusion_matrix": cm.astype(np.int64).tolist()}
