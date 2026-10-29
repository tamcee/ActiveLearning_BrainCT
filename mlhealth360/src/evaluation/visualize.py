"""Save simple CT / ground-truth / prediction overlays."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_overlay(image, target, prediction, output_path, title=""):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("Center CT")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(np.ma.masked_where(target == 0, target), cmap="tab10", vmin=0, vmax=5, alpha=.6)
    axes[1].set_title("Ground truth")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(np.ma.masked_where(prediction == 0, prediction), cmap="tab10", vmin=0, vmax=5, alpha=.6)
    axes[2].set_title("Prediction")
    for ax in axes: ax.axis("off")
    if title: fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
