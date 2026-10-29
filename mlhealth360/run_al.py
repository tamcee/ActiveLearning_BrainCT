"""Minimal AL scoring/filtering/selection orchestration.

This script deliberately stops at a selected-slice manifest. It does NOT
automatically reveal masks or train on the pool. Review the manifest and use
a controlled simulation step to reveal ground truth after selection.
"""
import argparse
from pathlib import Path
import pandas as pd
from src.active_learning.filtering import filter_candidates
from src.active_learning.selection import stratified_entropy_selection

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scores",default="experiments/active_learning/scores.csv")
    p.add_argument("--output",default="experiments/active_learning/selected_for_training.csv")
    p.add_argument("--budget",type=int,default=150)
    p.add_argument("--min-entropy",type=float,default=.10)
    p.add_argument("--min-pred-area",type=int,default=50)
    p.add_argument("--study-cap",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    scores=pd.read_csv(args.scores)
    eligible=filter_candidates(scores,args.min_entropy,args.min_pred_area)
    selected=stratified_entropy_selection(eligible,budget=args.budget,study_cap=args.study_cap,seed=args.seed)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);selected.to_csv(out,index=False)
    print(f"Candidates: {len(scores)}; eligible: {len(eligible)}; selected: {len(selected)}")
    print("Saved selection manifest:",out)
    print("Next: reveal masks only for selected samples in a controlled simulation; keep the remaining pool hidden.")

if __name__=="__main__": main()
