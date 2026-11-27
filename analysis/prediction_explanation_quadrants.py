"""Sample-level prediction/explanation quadrants from existing run artifacts.

Uses decision-margin SHAP (class-1 logit minus class-0 logit) instead of
unidentifiable individual class-logit attributions. No retraining required.

Run: python analysis/prediction_explanation_quadrants.py --run-dir runs/adult_multiplicity_seed1
Produces a sample table and threshold-sensitivity summary under --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_margin_shap(run_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    manifest = json.loads((run_dir / "shap_manifest.json").read_text())
    models = [str(x) for x in manifest["selected_model_ids"]]
    samples = np.asarray(manifest["explain_indices"], dtype=int)
    features = pd.read_csv(run_dir / "feature_names.csv")["feature"].astype(str).tolist()
    classes = pd.read_csv(run_dir / "class_names.csv")["class_name"].astype(str).tolist()
    if len(classes) != 2:
        raise ValueError("This analysis supports binary classification only")
    if len(models) < 2 or len(samples) < 2 or len(features) < 1:
        raise ValueError("Need >=2 models, >=2 explained samples, >=1 feature")
    if len(set(models)) != len(models) or len(set(samples)) != len(samples):
        raise ValueError("Duplicate models or explained samples")

    model_idx = {x: i for i, x in enumerate(models)}
    sample_idx = {int(x): i for i, x in enumerate(samples)}
    feature_idx = {x: i for i, x in enumerate(features)}
    class_idx = {x: i for i, x in enumerate(classes)}
    arr = np.full((len(models), len(samples), len(features), 2), np.nan, dtype=np.float32)
    with gzip.open(run_dir / "shap_values_long.csv.gz", "rt", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            m, s, f, c = row["model_id"], int(row["sample_id"]), row["feature"], row["class_name"]
            if m not in model_idx or s not in sample_idx:
                continue
            if f not in feature_idx or c not in class_idx:
                raise ValueError(f"Unexpected feature/class in SHAP row: {f}/{c}")
            arr[model_idx[m], sample_idx[s], feature_idx[f], class_idx[c]] = float(row["shap_value"])
    if not np.isfinite(arr).all():
        raise ValueError("Incomplete/non-finite SHAP cube: check model IDs and sample IDs")
    return models, samples, arr[..., 1] - arr[..., 0]


def load_predictions(run_dir: Path, models: list[str], samples: np.ndarray) -> np.ndarray:
    mi={x: i for i,x in enumerate(models)}
    si={int(x): i for i,x in enumerate(samples)}
    pred=np.full((len(models),len(samples)), -1, dtype=np.int16)
    for chunk in pd.read_csv(run_dir / "test_predictions.csv",
                             usecols=["model_id", "sample_id", "y_pred"],
                             chunksize=200_000):
        keep=chunk.loc[chunk["model_id"].isin(mi) & chunk["sample_id"].isin(si)]
        for row in keep.itertuples(index=False):
            value=int(row.y_pred)
            if value not in (0,1):
                raise ValueError("Non-binary prediction encountered")
            pred[mi[str(row.model_id)],si[int(row.sample_id)]]=value
    if (pred<0).any():
        raise ValueError("Some model/sample test predictions were not found")
    return pred


def mean_pairwise_cosine_distance(matrix: np.ndarray) -> float:
    matrix=np.asarray(matrix,dtype=np.float64)
    norms=np.linalg.norm(matrix,axis=1)
    nonzero=norms>1e-12
    safe=matrix / np.where(nonzero,norms,1.0)[:,None]
    cosine=np.clip(safe@safe.T,-1.,1.)
    dist=1.-cosine
    # Both-zero explanations are identical; one-zero vs nonzero is maximally uninformative.
    zero_pair=(~nonzero[:,None]) & (~nonzero[None,:])
    dist[zero_pair]=0.
    upper=dist[np.triu_indices(len(matrix),1)]
    return float(np.mean(upper))


def analyze(run_dir: Path, out_dir: Path, thresholds: tuple[float,...]=(0.05,0.10,0.15)) -> pd.DataFrame:
    models,samples,margin=load_margin_shap(run_dir)
    preds=load_predictions(run_dir, models, samples)
    n_models=len(models)
    n_pairs=n_models*(n_models-1)/2
    rows=[]
    for j,sid in enumerate(samples):
        votes=preds[:,j]
        n0=int((votes==0).sum()); n1=n_models-n0
        rows.append({
            "sample_id": int(sid),
            "n_models": n_models,
            "prediction_consensus": bool(n0==0 or n1==0),
            "pairwise_prediction_disagreement": n0*n1/n_pairs,
            "mean_pairwise_margin_shap_cosine_distance": mean_pairwise_cosine_distance(margin[:,j,:]),
        })
    table=pd.DataFrame(rows)
    out_dir.mkdir(parents=True,exist_ok=True)
    table.to_csv(out_dir/"quadrant_sample_metrics.csv",index=False)
    summaries=[]
    for t in thresholds:
        same_p=table["prediction_consensus"].to_numpy()
        same_e=(table["mean_pairwise_margin_shap_cosine_distance"].to_numpy()<=t)
        summaries.append({
            "run_dir": str(run_dir), "n_samples":len(table),"n_models":n_models,
            "explanation_distance_threshold":t,
            "prediction_same_explanation_same":float(np.mean(same_p & same_e)),
            "prediction_same_explanation_different":float(np.mean(same_p & ~same_e)),
            "prediction_different_explanation_same":float(np.mean(~same_p & same_e)),
            "prediction_different_explanation_different":float(np.mean(~same_p & ~same_e)),
            "spearman_conflict_vs_explanation_distance":float(table["pairwise_prediction_disagreement"].corr(
                table["mean_pairwise_margin_shap_cosine_distance"],method="spearman")),
        })
    result=pd.DataFrame(summaries)
    result.to_csv(out_dir/"quadrant_threshold_sensitivity.csv",index=False)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--thresholds",type=float,nargs="+",default=[.05,.10,.15])
    args=parser.parse_args()
    target=args.out_dir or args.run_dir/"quadrant_analysis"
    print(analyze(args.run_dir,target,tuple(args.thresholds)).to_string(index=False))


if __name__=="__main__":
    main()
