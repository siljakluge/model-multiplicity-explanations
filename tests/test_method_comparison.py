"""Offline fixtures for group-aligned attribution comparisons and cache repair."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mmcyber.method_comparison import (
    group_feature_space, group_attributions, original_lime_matrix,
    lime_forward_transform,
)
from analysis.compare_methods import analyze_methods
from analysis.rebuild_shap_summary import rebuild


class MultiMethodTests(unittest.TestCase):
    def test_original_feature_mapping_reconstructs_valid_categorical_rows(self):
        raw = pd.DataFrame({
            'age': [20., 40., 60., 50.],
            'job': ['a', 'b', 'c', 'a'],
            'marital': ['single', 'married', 'married', 'single'],
        })
        preprocess = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), ['age']),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
             ['job', 'marital']),
        ])
        x = preprocess.fit_transform(raw).astype(np.float32)
        space = group_feature_space(preprocess, list(preprocess.get_feature_names_out()))
        self.assertEqual(space.names, ['age', 'job', 'marital'])
        self.assertEqual([len(idx) for idx in space.indices], [1, 3, 2])
        decoded = original_lime_matrix(x, preprocess, space)
        restored = lime_forward_transform(decoded, preprocess, space)
        np.testing.assert_allclose(restored, x, atol=1e-5)
        vals = np.arange(x.shape[1], dtype=float)
        np.testing.assert_allclose(group_attributions(vals, space), [0., 6., 9.])
        # Every categorical perturbation is exactly one category, not 2+ hot.
        changed = decoded[:1].copy()
        changed[0, 1] = 2  # alternate valid job category
        xx = lime_forward_transform(changed, preprocess, space)
        self.assertAlmostEqual(float(xx[0, 1:4].sum()), 1.0)
        # One-hot unknowns on held-out samples must remain all-zero, not imputed.
        unknown = decoded[:1].copy()
        unknown[0, 1] = len(space.cat_categories[0])
        unknown_encoded = lime_forward_transform(unknown, preprocess, space)
        self.assertAlmostEqual(float(unknown_encoded[0, 1:4].sum()), 0.0)

    def test_method_quadrants_are_sample_level(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            out = run / 'method_comparison'
            out.mkdir()
            models = ['m1', 'm2']
            samples = [0, 1]
            feats = ['a', 'b']
            (out / 'manifest.json').write_text(json.dumps({
                'models': models, 'samples': samples,
                'original_feature_names': feats, 'methods': ['shap', 'ig'],
            }))
            pd.DataFrame([
                {'model_id': m, 'sample_id': s,
                 'y_pred': (1 if s == 0 else int(m == 'm1'))}
                for m in models for s in samples
            ]).to_csv(run / 'test_predictions.csv', index=False)
            rows = []
            for method in ['shap', 'ig']:
                for m in models:
                    for s in samples:
                        vector = np.asarray([1., 0.]) if s == 0 or m == 'm1' else np.asarray([-1., 0.])
                        for f, a in zip(feats, vector):
                            rows.append({'method': method, 'model_id': m, 'sample_id': s,
                                         'feature': f, 'attribution': a})
            pd.DataFrame(rows).to_csv(out / 'attributions_long.csv.gz', index=False, compression='gzip')
            result = analyze_methods(run, (0.1,))
            self.assertEqual(len(result), 2)
            for row in result.itertuples(index=False):
                self.assertAlmostEqual(row.both_same, .5)
                self.assertAlmostEqual(row.both_different, .5)
                self.assertAlmostEqual(row.prediction_same_explanation_different, 0.)

    def test_rebuild_full_feature_shap_from_npz_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run/'shap_values').mkdir()
            models = ['m1', 'm2']
            feat = ['f1', 'f2', 'f3', 'f4']
            classes = ['negative', 'positive']
            pd.DataFrame({'model_id': models}).to_csv(run/'rashomon_models.csv', index=False)
            pd.DataFrame({'feature': feat}).to_csv(run/'feature_names.csv', index=False)
            old = []
            for model in models:
                for cls in classes:
                    old.append({'model_id': model, 'class_name': cls, 'feature': 'f1',
                                'rank': 1, 'mean_abs_shap': 1., 'activation': 'relu'})
                np.savez_compressed(run/'shap_values'/f'{model}.npz',
                                    shap_values=np.random.default_rng(12).normal(size=(2, 3, 4)),
                                    feature_names=np.array(feat), class_names=np.array(classes))
            pd.DataFrame(old).to_csv(run/'shap_summary.csv', index=False)
            self.assertEqual(rebuild(run), 2*2*4)
            output = pd.read_csv(run/'shap_summary.csv')
            self.assertEqual(output.groupby(['model_id', 'class_name']).size().unique().tolist(), [4])
            self.assertTrue((run/'shap_summary_before_full_rebuild.csv').is_file())


if __name__ == '__main__':
    unittest.main()
