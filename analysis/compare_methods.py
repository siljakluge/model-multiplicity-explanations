"""Compare prediction/explanation agreement across SHAP, IG, and LIME.

Reads the grouped original-feature attributions created by
``python -m mmcyber.method_comparison`` and the saved model predictions.
Reports continuous associations AND thresholded four-way categories.

Samples and models are taken from method_comparison/manifest.json. All reported
percentages are sample-level, not independent model-pair observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.prediction_explanation_quadrants import (
    load_predictions, mean_pairwise_cosine_distance,
)


def analyze_methods(run_dir: str | Path, thresholds=(0.05, 0.10, 0.15)) -> pd.DataFrame:
    run_dir = Path(run_dir)
    output = run_dir / 'method_comparison'
    manifest = json.loads((output / 'manifest.json').read_text())
    models = list(map(str, manifest['models']))
    samples = np.asarray(manifest['samples'], dtype=int)
    features = list(map(str, manifest['original_feature_names']))
    methods = list(map(str, manifest['methods']))
    if len(models) < 2 or len(samples) < 1 or not features:
        raise ValueError('Invalid method comparison selection')
    mi = {m: i for i, m in enumerate(models)}
    si = {int(s): i for i, s in enumerate(samples)}
    fi = {f: i for i, f in enumerate(features)}
    cube = {method: np.full((len(models), len(samples), len(features)), np.nan, dtype=np.float32)
            for method in methods}
    expected_rows = len(models) * len(samples) * len(features) * len(methods)
    observed = 0
    for chunk in pd.read_csv(output / 'attributions_long.csv.gz', chunksize=200_000):
        if not {'method', 'model_id', 'sample_id', 'feature', 'attribution'} <= set(chunk.columns):
            raise ValueError('Malformed method_attributions file')
        for row in chunk.itertuples(index=False):
            method, model_id, sample_id, feature, value = (str(row.method), str(row.model_id),
                                                           int(row.sample_id), str(row.feature),
                                                           float(row.attribution))
            if method not in cube or model_id not in mi or sample_id not in si or feature not in fi:
                raise ValueError('Unexpected attribution identity')
            x = cube[method]
            index = (mi[model_id], si[sample_id], fi[feature])
            if not np.isnan(x[index]):
                raise ValueError('Duplicated attribution row')
            x[index] = value
            observed += 1
    if observed != expected_rows or any(not np.isfinite(x).all() for x in cube.values()):
        raise ValueError(f'Incomplete/nonfinite attribution data ({observed}/{expected_rows})')

    predictions = load_predictions(run_dir, models, samples)
    n_models = len(models)
    n_pairs = n_models * (n_models - 1) / 2
    n_positive = (predictions == 1).sum(axis=0)
    prediction_same = (n_positive == 0) | (n_positive == n_models)
    pairwise_disagreement = (n_models - n_positive) * n_positive / n_pairs

    per_sample = []
    summary = []
    for method, values in cube.items():
        distances = np.asarray([
            mean_pairwise_cosine_distance(values[:, i, :])
            for i in range(len(samples))
        ])
        for sample, agree, conflict, d in zip(samples, prediction_same, pairwise_disagreement, distances):
            per_sample.append({
                'method': method, 'sample_id': int(sample),
                'prediction_same': bool(agree), 'pairwise_prediction_disagreement': float(conflict),
                'mean_pairwise_explanation_cosine_distance': float(d),
            })
        correlation = pd.Series(pairwise_disagreement).corr(pd.Series(distances), method='spearman')
        for threshold in thresholds:
            explanation_same = distances <= threshold
            summary.append({
                'method': method, 'n_models': n_models, 'n_samples': len(samples),
                'explanation_distance_threshold': float(threshold),
                'both_same': float(np.mean(prediction_same & explanation_same)),
                'prediction_same_explanation_different': float(np.mean(prediction_same & ~explanation_same)),
                'prediction_different_explanation_same': float(np.mean(~prediction_same & explanation_same)),
                'both_different': float(np.mean(~prediction_same & ~explanation_same)),
                'mean_explanation_distance': float(np.mean(distances)),
                'spearman_prediction_disagreement_vs_explanation_distance': float(correlation),
            })
    pd.DataFrame(per_sample).to_csv(output / 'per_sample_agreement.csv', index=False)
    result = pd.DataFrame(summary)
    result.to_csv(output / 'quadrants_by_method.csv', index=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.10, 0.15])
    args = parser.parse_args()
    print(analyze_methods(args.run_dir, args.thresholds).to_string(index=False))
