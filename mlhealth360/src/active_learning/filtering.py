"""Candidate quality filters for AL slice scores."""
import numpy as np
import pandas as pd

def filter_candidates(scores: pd.DataFrame, min_entropy=0.10, min_pred_area=50):
    required = {"entropy", "predicted_foreground_pixels"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Score table missing columns: {sorted(missing)}")
    eligible = scores[
        (scores["entropy"] >= float(min_entropy)) &
        (scores["predicted_foreground_pixels"] >= int(min_pred_area))
    ].copy()
    return eligible.reset_index(drop=True)
