"""Offline regression tests for the new benchmark loaders."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from mmcyber.data import prepare_dataset
from mmcyber.extra_datasets import GERMAN_FEATURES, GERMAN_NUMERIC


class ExtraDatasetTests(unittest.TestCase):
    def test_german_credit_exact_binary_labels_no_data_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/"raw"; raw.mkdir()
            vals = []
            for i in range(120):
                row = {col: f"A{i % 4 + 11}" for col in GERMAN_FEATURES}
                for col in GERMAN_NUMERIC:
                    row[col] = 10 + i % 37
                row["class"] = 2 if i % 4 == 0 else 1
                vals.append(row)
            pd.DataFrame(vals).to_csv(raw/"german.data", sep=" ", header=False, index=False)
            cfg={"dataset":{"name":"german_credit","raw_dir":str(raw),"validate_full_size":False,"split_seed":17}}
            d1=prepare_dataset(cfg,Path(tmp)/"one")
            d2=prepare_dataset(cfg,Path(tmp)/"two")
            self.assertEqual(d1.class_names,["good","bad"])
            self.assertEqual(len(d1.y_train)+len(d1.y_val)+len(d1.y_test),120)
            np.testing.assert_array_equal(d1.y_test,d2.y_test)
            self.assertTrue(np.isfinite(d1.x_test).all())
            self.assertFalse(any("class" in f for f in d1.feature_names))

    def test_compas_excludes_scores_and_outcome_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/"raw";raw.mkdir()
            vals=[]
            for i in range(160):
                vals.append(dict(age=20+i%40,sex="Female" if i%2 else "Male",race="Caucasian" if i%3 else "African-American",
                    juv_fel_count=i%3,juv_misd_count=0,juv_other_count=1,priors_count=i%6,c_charge_degree="F" if i%2 else "M",
                    two_year_recid=i%2,is_recid=i%2,score_text="High",decile_score=9,days_b_screening_arrest=-2))
            vals[0]["days_b_screening_arrest"]=90
            pd.DataFrame(vals).to_csv(raw/"compas-scores-two-years.csv",index=False)
            cfg={"dataset":{"name":"compas","raw_dir":str(raw),"validate_full_size":False,"split_seed":7}}
            d=prepare_dataset(cfg,Path(tmp)/"out")
            self.assertEqual(len(d.y_train)+len(d.y_val)+len(d.y_test),159)
            self.assertEqual(d.class_names,["no_recidivism","recidivism"])
            self.assertFalse(any("score" in s or "recid" in s or "days_b_" in s for s in d.feature_names))
            self.assertTrue(np.isfinite(d.x_train).all())

    def test_taiwan_drop_id_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/"raw";raw.mkdir()
            (raw/"default of credit card clients.xls").write_bytes(b"mocked reader")
            values={"ID":list(range(140)),"LIMIT_BAL":[10000+i*4 for i in range(140)],
                    "SEX":[i%2+1 for i in range(140)],"EDUCATION":[i%3+1 for i in range(140)],
                    "MARRIAGE":[i%2+1 for i in range(140)],"AGE":[23+i%40 for i in range(140)]}
            for i in range(1,7):
                values[f"PAY_{0 if i == 1 else i}"]=[j%4 for j in range(140)]
                values[f"BILL_AMT{i}"]=[j*9 for j in range(140)]
                values[f"PAY_AMT{i}"]=[j*3 for j in range(140)]
            values["default payment next month"]=[i%2 for i in range(140)]
            frame=pd.DataFrame(values)
            cfg={"dataset":{"name":"taiwan_credit","raw_dir":str(raw),"validate_full_size":False}}
            with patch("mmcyber.extra_datasets.pd.read_excel",return_value=frame):
                d=prepare_dataset(cfg,Path(tmp)/"out")
            self.assertEqual(d.class_names,["no_default","default"])
            self.assertFalse(any("ID" in s or "default" in s for s in d.feature_names))
            self.assertGreater(d.x_train.shape[1],23)

    def test_breast_cancer_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={"dataset":{"name":"breast_cancer","split_seed":42}}
            d=prepare_dataset(cfg,Path(tmp)/"out")
            self.assertEqual(len(d.y_train)+len(d.y_val)+len(d.y_test),569)
            self.assertEqual(d.class_names,["malignant","benign"])
            self.assertEqual(d.x_train.shape[1],30)
            self.assertTrue(np.isfinite(d.x_val).all())


if __name__=='__main__':
    unittest.main()
