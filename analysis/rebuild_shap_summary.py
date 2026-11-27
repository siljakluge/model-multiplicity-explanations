"""Reconstruct *full-feature* DeepSHAP importance ranks from cached NPZ values.

Use on historical runs where shap_summary.csv was truncated to 50 features.
No model inference, training or explanation recomputation required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rebuild(run_dir: str | Path) -> int:
    run = Path(run_dir)
    inventory = pd.read_csv(run / 'feature_names.csv')['feature'].astype(str).tolist()
    membership = pd.read_csv(run / 'rashomon_models.csv')
    old = pd.read_csv(run / 'shap_summary.csv')
    rows = []
    for model_id in membership['model_id'].astype(str):
        path = run / 'shap_values' / f'{model_id}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing cached SHAP arrays: {path}')
        with np.load(path) as saved:
            values = np.asarray(saved['shap_values'])
            features = list(map(str, saved['feature_names']))
            classes = list(map(str, saved['class_names']))
        if features != inventory or values.ndim != 3 or values.shape[0] != len(classes) or values.shape[2] != len(features):
            raise ValueError(f'Unexpected SHAP shape/features in {path}: {values.shape}')
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite SHAP values in {path}')
        meta_rows = old[old['model_id'].astype(str) == model_id]
        if meta_rows.empty:
            raise ValueError(f'Original SHAP metadata not found for {model_id}')
        metadata = {
            name: meta_rows.iloc[0][name]
            for name in ('architecture_id', 'hidden_dims_label', 'activation', 'seed', 'subset_fraction')
            if name in old.columns
        }
        for idx, class_name in enumerate(classes):
            mean_abs = np.abs(values[idx]).mean(axis=0)
            for rank, feature_idx in enumerate(np.argsort(-mean_abs, kind='stable'), start=1):
                rows.append({
                    'model_id': model_id, **metadata, 'class_name': class_name,
                    'rank': int(rank), 'feature': features[feature_idx],
                    'mean_abs_shap': float(mean_abs[feature_idx]),
                })
    expected_rows = len(membership) * len(inventory) * 2
    if len(rows) != expected_rows:
        raise ValueError(f'Unexpected full-ranking row count: {len(rows)} != {expected_rows}')
    target = run / 'shap_summary.csv'
    old.to_csv(run / 'shap_summary_before_full_rebuild.csv', index=False)
    pd.DataFrame(rows).to_csv(target, index=False)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(f'Rebuilt {rebuild(args.run_dir)} complete SHAP importance records')
