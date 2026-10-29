"""Study-aware and entropy-stratified slice selection."""
import pandas as pd

def study_aware_topk(scores, k_per_study=5, study_cap=2, total_budget=150,
                     score_column="entropy"):
    required = {"study", "slice", score_column}
    if required - set(scores.columns):
        raise ValueError(f"Scores need columns {sorted(required)}")
    candidates = scores.sort_values(score_column, ascending=False)
    candidates = candidates.groupby("study", group_keys=False).head(k_per_study)
    candidates = candidates.sort_values(score_column, ascending=False)
    selected, counts = [], {}
    for _, row in candidates.iterrows():
        study = row["study"]
        if counts.get(study, 0) >= study_cap:
            continue
        selected.append(row)
        counts[study] = counts.get(study, 0) + 1
        if len(selected) >= total_budget:
            break
    return pd.DataFrame(selected).reset_index(drop=True)

def stratified_entropy_selection(scores, budget=150, bands=None, study_cap=2, seed=42):
    """
    Default SRS bands: 40 high (>=.50), 40 medium [.30,.50), 70 low [.10,.30).
    If a band has fewer eligible candidates, unused quota is filled from remaining
    eligible candidates by entropy rank while preserving the per-study cap.
    """
    if bands is None:
        bands = [("high", 0.50, float("inf"), 40),
                 ("medium", 0.30, 0.50, 40),
                 ("low", 0.10, 0.30, 70)]
    required = {"study", "slice", "entropy"}
    if required - set(scores.columns):
        raise ValueError(f"Scores need columns {sorted(required)}")
    df = scores.copy()
    rng = __import__("numpy").random.default_rng(seed)
    df["_tie"] = rng.random(len(df))
    selected_indices, counts = [], {}
    for _, low, high, quota in bands:
        band = df[(df["entropy"] >= low) & (df["entropy"] < high)].sort_values(
            ["entropy", "_tie"], ascending=[False, True])
        for idx, row in band.iterrows():
            if len([i for i in selected_indices if df.loc[i, "entropy"] >= low and df.loc[i, "entropy"] < high]) >= quota:
                break
            study = row["study"]
            if counts.get(study, 0) >= study_cap:
                continue
            selected_indices.append(idx)
            counts[study] = counts.get(study, 0) + 1
    if len(selected_indices) < budget:
        remainder = df.drop(index=selected_indices).sort_values("entropy", ascending=False)
        for idx, row in remainder.iterrows():
            study = row["study"]
            if counts.get(study, 0) >= study_cap:
                continue
            selected_indices.append(idx)
            counts[study] = counts.get(study, 0) + 1
            if len(selected_indices) >= budget:
                break
    out = df.loc[selected_indices].drop(columns=["_tie"], errors="ignore").copy()
    out["selection_order"] = range(1, len(out)+1)
    return out.head(budget).reset_index(drop=True)
