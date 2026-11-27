"""Optional decision-margin explanations from SHAP, Integrated Gradients and LIME.

Run after ``mmcyber.cli shap``. All methods explain logit_1 - logit_0 for
exactly the SAME selected model IDs and test sample IDs. SHAP / IG dummy
attributions are summed to their original categorical features, while LIME
is fitted in the ORIGINAL feature space, avoiding invalid multi-hot samples.

Important: LIME regression coefficients and Shapley / IG attributions are not
identical estimands; comparisons are descriptive and require method-level
stochasticity / fidelity diagnostics.
"""
from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd


@dataclass
class FeatureSpace:
    names: list[str]
    indices: list[np.ndarray]
    num_names: list[str]
    cat_names: list[str]
    cat_categories: list[list[object]]


def group_feature_space(preprocessor, transformed_names: list[str]) -> FeatureSpace:
    """Map preprocessor columns to original tabular variables in output order.

    Works with the project's ColumnTransformer: num (impute + scale),
    cat (impute + onehot, dense). Fails closed on unfamiliar transformations.
    """
    num_columns = list(preprocessor.transformers_[0][2])
    cat_columns = list(preprocessor.transformers_[1][2])
    if preprocessor.transformers_[0][0] != 'num' or preprocessor.transformers_[1][0] != 'cat':
        raise ValueError('Expected num/cat ColumnTransformer blocks in existing repo')
    categories = []
    if cat_columns:
        encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categories = [list(values) for values in encoder.categories_]
    groups = []
    start = 0
    for _ in num_columns:
        groups.append(np.asarray([start], dtype=int))
        start += 1
    for values in categories:
        groups.append(np.arange(start, start + len(values), dtype=int))
        start += len(values)
    if start != len(transformed_names) or len(groups) != len(num_columns) + len(cat_columns):
        raise ValueError(f'Feature-group dimension mismatch {start} vs {len(transformed_names)}')
    expected = list(preprocessor.get_feature_names_out())
    if expected != transformed_names:
        raise ValueError('Saved feature inventory differs from fitted preprocessor')
    return FeatureSpace(num_columns + cat_columns, groups, num_columns, cat_columns, categories)


def group_attributions(values: np.ndarray, space: FeatureSpace) -> np.ndarray:
    """Sum contributions of one-hot dummies to the original feature group."""
    values = np.asarray(values)
    return np.stack([values[..., indices].sum(axis=-1) for indices in space.indices], axis=-1)


def original_lime_matrix(x: np.ndarray, preprocessor, space: FeatureSpace) -> np.ndarray:
    """Invert the fitted project's preprocessing to LIME-ready raw values.

    Missing train values were already imputed in the original transform; unseen
    test categories are assigned an extra sentinel integer code.
    """
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != sum(map(len, space.indices)):
        raise ValueError('Unexpected transformed matrix shape')
    n_num = len(space.num_names)
    parts = []
    if n_num:
        scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        parts.append(np.asarray(scaler.inverse_transform(x[:, :n_num]), dtype=np.float64))
    if space.cat_names:
        cat_values = preprocessor.named_transformers_['cat'].named_steps['onehot'].inverse_transform(x[:, n_num:])
        for i, cats in enumerate(space.cat_categories):
            lut = {str(value): j for j, value in enumerate(cats)}
            indices = [lut.get(str(value), len(cats)) for value in cat_values[:, i]]
            parts.append(np.asarray(indices, dtype=np.float64)[:, None])
    return np.concatenate(parts, axis=1)


def lime_forward_transform(x: np.ndarray, preprocessor, space: FeatureSpace) -> np.ndarray:
    """Re-encode LIME perturbations respecting each original categorical group."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(space.names):
        raise ValueError('Bad LIME raw input shape')
    frame = pd.DataFrame(index=range(len(x)))
    n_num = len(space.num_names)
    for i, name in enumerate(space.num_names):
        frame[name] = x[:, i]
    for i, (name, cats) in enumerate(zip(space.cat_names, space.cat_categories)):
        vals = np.rint(x[:, n_num + i]).astype(int)
        frame[name] = pd.Series([cats[j] if 0 <= j < len(cats) else '__LIME_UNSEEN_CATEGORY__' for j in vals], dtype=object)
    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    if transformed.shape[1] != sum(map(len, space.indices)):
        raise ValueError('LIME forward transform produced the wrong dimension')
    return transformed


def _read_context(run_dir: Path):
    manifest_path = run_dir / 'shap_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Run mmcyber.cli shap before multi-explain to align samples')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    models = list(map(str, manifest['selected_model_ids']))
    samples = np.asarray(manifest['explain_indices'], dtype=int)
    if len(models) < 2 or len(samples) < 1:
        raise ValueError('At least two models and one explained sample required')
    if len(set(models)) != len(models) or len(set(samples)) != len(samples):
        raise ValueError('Duplicate model/sample identifiers')
    cache_file = run_dir / 'shap_dataset.joblib'
    if cache_file.is_file():
        data = joblib.load(cache_file)
        x_train, x_test = np.asarray(data['x_train']), np.asarray(data['x_test'])
        features = list(data['feature_names'])
        classes = list(data['class_names'])
    else:
        from mmcyber.data import prepare_dataset
        from mmcyber.utils import load_config
        prepared = prepare_dataset(load_config(run_dir / 'config.resolved.json'), run_dir)
        x_train, x_test = prepared.x_train, prepared.x_test
        features, classes = prepared.feature_names, prepared.class_names
    if len(classes) != 2:
        raise ValueError('Only binary classification is supported for margin explanations')
    if (samples < 0).any() or (samples >= len(x_test)).any():
        raise ValueError('SHAP sample indices are out of range')
    if list(pd.read_csv(run_dir / 'feature_names.csv')['feature']) != features:
        raise ValueError('Feature inventory mismatch')
    pre = joblib.load(run_dir / 'preprocessor.joblib')
    space = group_feature_space(pre, features)
    return models, samples, x_train, x_test, pre, space


def _load_shap_grouped(run_dir: Path, models: list[str], samples: np.ndarray, space: FeatureSpace) -> np.ndarray:
    from analysis.prediction_explanation_quadrants import load_margin_shap
    full_models, full_samples, transformed = load_margin_shap(run_dir)
    model_idx = {m: j for j, m in enumerate(full_models)}
    sample_idx = {int(s): j for j, s in enumerate(full_samples)}
    indices_models = [model_idx[m] for m in models]
    indices_samples = [sample_idx[int(s)] for s in samples]
    selected = transformed[np.ix_(indices_models, indices_samples, range(transformed.shape[2]))]
    return group_attributions(selected, space).astype(np.float32)


def compute_method_comparison(
    run_dir: str | Path, *, methods: tuple[str, ...] = ('shap', 'ig', 'lime'),
    max_models: int | None = None, max_samples: int | None = 64,
    device: str = 'cpu', ig_steps: int = 64, ig_batch: int = 16,
    lime_samples: int = 1000, lime_reference_size: int = 2000,
    random_state: int = 42, overwrite: bool = False,
) -> Path:
    """Write one reproducible long-form original-feature attribution table.

    The 'shap' method uses EXISTING DeepSHAP logits and computes their difference;
    no additional DeepSHAP work is needed. Other methods are optional.
    """
    run_dir = Path(run_dir)
    methods = tuple(dict.fromkeys(methods))
    if not methods or not set(methods) <= {'shap', 'ig', 'lime'}:
        raise ValueError('methods must be a nonempty subset of shap, ig, lime')
    if ig_steps < 2 or ig_batch < 1 or lime_samples < 50 or lime_reference_size < 10:
        raise ValueError('Invalid IG or LIME sampling parameters')
    models, samples, x_train, x_test, pre, space = _read_context(run_dir)
    models = models[:max_models] if max_models is not None else models
    samples = samples[:max_samples] if max_samples is not None else samples
    if len(models) < 2 or not len(samples):
        raise ValueError('Selection left fewer than two models or zero samples')
    output = run_dir / 'method_comparison'
    output.mkdir(parents=True, exist_ok=True)
    params = {
        'methods': list(methods), 'models': models, 'samples': samples.tolist(),
        'original_feature_names': space.names, 'source': 'logit_1-minus-logit_0',
        'ig_steps': ig_steps, 'ig_batch': ig_batch,
        'lime_samples': lime_samples, 'lime_reference_size': lime_reference_size,
        'random_state': random_state, 'device': device,
        'preprocessing': 'original-feature groups; categorical one-hot blocks summed for SHAP/IG',
    }
    params_path = output / 'manifest.json'
    if params_path.exists() and not overwrite:
        if params == json.loads(params_path.read_text()) and (output / 'attributions_long.csv.gz').exists():
            return output
        raise FileExistsError('Previous method comparison exists with different settings; use --overwrite')

    # Validate method dependencies BEFORE opening output files to avoid partial results.
    if 'ig' in methods:
        try:
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise ImportError("IG needs Captum; install: pip install -e '.[explain]'") from exc
    if 'lime' in methods:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:
            raise ImportError("LIME needs lime; install: pip install -e '.[explain]'") from exc

    import torch
    from mmcyber.model import resolve_device
    from mmcyber.shap_worker import _load_torch_model

    device_obj = resolve_device(device)
    diagnostics = []
    long_file = output / 'attributions_long.csv.gz'
    temporary_file = output / 'attributions_long.csv.gz.part'
    field_names = ['method', 'model_id', 'sample_id', 'feature', 'attribution']

    shap_attributions = _load_shap_grouped(run_dir, models, samples, space) if 'shap' in methods else None
    if shap_attributions is not None and not np.isfinite(shap_attributions).all():
        raise ValueError('Nonfinite SHAP margin attributions')

    raw_train = raw_test = None
    lime_background = None
    if 'lime' in methods:
        raw_train = original_lime_matrix(x_train, pre, space)
        raw_test = original_lime_matrix(x_test[samples], pre, space)
        rng = np.random.default_rng(random_state)
        subset = rng.choice(len(raw_train), min(len(raw_train), lime_reference_size), replace=False)
        lime_background = raw_train[subset]

    with gzip.open(temporary_file, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for model_index, model_id in enumerate(models):
            model = None
            if 'ig' in methods or 'lime' in methods:
                path = run_dir / 'models' / f'{model_id}.pt'
                if not path.exists():
                    raise FileNotFoundError(f'Missing checkpoint {path}')
                model = _load_torch_model(path, device_obj)
                model.eval()

            def write_attributions(method: str, vals: np.ndarray):
                vals = np.asarray(vals)
                if vals.shape != (len(samples), len(space.names)) or not np.isfinite(vals).all():
                    raise ValueError(f'Bad explanation shape/values {method}: {vals.shape}')
                for i, sample in enumerate(samples):
                    writer.writerows({
                        'method': method, 'model_id': model_id,
                        'sample_id': int(sample), 'feature': feat,
                        'attribution': float(value),
                    } for feat, value in zip(space.names, vals[i]))

            if shap_attributions is not None:
                write_attributions('shap', shap_attributions[model_index])

            if 'ig' in methods:
                ig = IntegratedGradients(lambda x: model(x)[:, 1] - model(x)[:, 0])
                baseline = torch.from_numpy(x_train.mean(axis=0, dtype=np.float64).astype(np.float32)[None, :]).to(device_obj)
                attrs = []
                for start in range(0, len(samples), ig_batch):
                    batch = torch.from_numpy(np.asarray(x_test[samples[start:start + ig_batch]], dtype=np.float32)).to(device_obj)
                    attributions, delta = ig.attribute(
                        batch, baselines=baseline, n_steps=ig_steps,
                        internal_batch_size=max(ig_steps, 128),
                        return_convergence_delta=True,
                    )
                    attrs.append(attributions.detach().cpu().numpy())
                    for sid, residual in zip(samples[start:start + ig_batch], delta.detach().cpu().numpy()):
                        diagnostics.append({'method': 'ig', 'model_id': model_id,
                                            'sample_id': int(sid), 'metric': 'abs_completeness_delta',
                                            'value': float(abs(residual))})
                write_attributions('ig', group_attributions(np.concatenate(attrs, axis=0), space))

            if 'lime' in methods:
                # Original variables, not independently-sampled one-hot dummies.
                n_num = len(space.num_names)
                categorical_ids = list(range(n_num, len(space.names)))
                categorical_names = {n_num + j: [str(v) for v in cats] + ['<unknown>']
                                     for j, cats in enumerate(space.cat_categories)}
                explainer = LimeTabularExplainer(
                    training_data=lime_background, feature_names=space.names,
                    categorical_features=categorical_ids,
                    categorical_names=categorical_names,
                    mode='regression', discretize_continuous=False,
                    feature_selection='none', sample_around_instance=True,
                    random_state=random_state,
                )

                def predict_margin(raw_x):
                    transformed = lime_forward_transform(raw_x, pre, space)
                    with torch.no_grad():
                        return np.concatenate([
                            (out[:, 1] - out[:, 0]).cpu().numpy()
                            for out in (model(torch.from_numpy(transformed[i:i+4096]).to(device_obj))
                                        for i in range(0, len(transformed), 4096))
                        ])
                vectors = np.zeros((len(samples), len(space.names)), dtype=np.float32)
                for i, sid in enumerate(samples):
                    # Same local perturbation seed across model choices for a given sample.
                    explainer.random_state = np.random.RandomState(random_state + int(sid))
                    exp = explainer.explain_instance(
                        raw_test[i], predict_margin,
                        num_features=len(space.names), num_samples=lime_samples,
                    )
                    for feature_idx, coefficient in exp.local_exp[1]:
                        vectors[i, int(feature_idx)] = float(coefficient)
                    diagnostics.append({'method': 'lime', 'model_id': model_id,
                                        'sample_id': int(sid), 'metric': 'local_surrogate_r2',
                                        'value': float(exp.score[1])})
                write_attributions('lime', vectors)

    temporary_file.replace(long_file)
    pd.DataFrame(diagnostics, columns=['method', 'model_id', 'sample_id', 'metric', 'value']).to_csv(
        output / 'diagnostics.csv', index=False)
    params_path.write_text(json.dumps(params, indent=2) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--methods', nargs='+', choices=['shap', 'ig', 'lime'], default=['shap', 'ig', 'lime'])
    parser.add_argument('--max-models', type=int)
    parser.add_argument('--max-samples', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ig-steps', type=int, default=64)
    parser.add_argument('--ig-batch', type=int, default=16)
    parser.add_argument('--lime-samples', type=int, default=1000)
    parser.add_argument('--lime-reference-size', type=int, default=2000)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    options = parser.parse_args()
    print(compute_method_comparison(
        options.run_dir, methods=tuple(options.methods), max_models=options.max_models,
        max_samples=options.max_samples, device=options.device,
        ig_steps=options.ig_steps, ig_batch=options.ig_batch,
        lime_samples=options.lime_samples, lime_reference_size=options.lime_reference_size,
        random_state=options.random_state, overwrite=options.overwrite,
    ))
