"""Per-pixel entropy, BALD, variance and slice-level uncertainty aggregation."""
import torch
from .mc_dropout import mc_predict

EPS = 1e-8

def entropy(probabilities, normalized=False):
    h = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=-3)
    if normalized:
        h = h / torch.log(torch.tensor(float(probabilities.shape[-3]), device=probabilities.device))
    return h

@torch.no_grad()
def score_batch(model, images, passes=10, foreground_only=True):
    """
    Returns per-sample uncertainty scores and maps.
    images: (B,C,H,W); MC probs: (T,B,K,H,W).
    BALD = H(mean p) - mean(H(p_t)).
    """
    probs, variance = mc_predict(model, images, passes=passes)
    mean_probs = probs.mean(dim=0)
    predictive_entropy = entropy(mean_probs)                 # (B,H,W)
    expected_entropy = entropy(probs).mean(dim=0)             # (B,H,W)
    bald = (predictive_entropy - expected_entropy).clamp_min(0)
    variance_map = variance.mean(dim=1)                       # average class variance -> (B,H,W)
    pred = mean_probs.argmax(dim=1)
    scores = {}
    for name, heatmap in {
        "entropy": predictive_entropy,
        "bald": bald,
        "variance": variance_map,
    }.items():
        per_sample = []
        for i in range(images.shape[0]):
            region = (pred[i] > 0) if foreground_only else torch.ones_like(pred[i], dtype=torch.bool)
            # If model predicts no foreground, fall back to all pixels rather than NaN.
            if not region.any():
                region = torch.ones_like(region)
            per_sample.append(heatmap[i][region].mean())
        scores[name] = torch.stack(per_sample)
    return {"scores": scores, "mean_probs": mean_probs, "prediction": pred,
            "entropy_map": predictive_entropy, "bald_map": bald,
            "variance_map": variance_map}
