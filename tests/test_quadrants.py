from __future__ import annotations
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.prediction_explanation_quadrants import analyze, mean_pairwise_cosine_distance

class QuadrantTests(unittest.TestCase):
    def test_distances_and_binary_cases(self):
        self.assertAlmostEqual(mean_pairwise_cosine_distance(np.asarray([[0.,0.],[0.,0.]])),0.)
        self.assertAlmostEqual(mean_pairwise_cosine_distance(np.asarray([[1.,0.],[1.,0.]])),0.)
        self.assertAlmostEqual(mean_pairwise_cosine_distance(np.asarray([[1.,0.],[-1.,0.]])),2.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); models=['a','b','c']; samples=[0,1];features=['x','y'];classes=['negative','positive']
            pd.DataFrame({'feature':features}).to_csv(root/'feature_names.csv',index=False)
            pd.DataFrame({'class_name':classes}).to_csv(root/'class_names.csv',index=False)
            (root/'shap_manifest.json').write_text(json.dumps({'selected_model_ids':models,'explain_indices':samples}))
            with gzip.open(root/'shap_values_long.csv.gz','wt',newline='') as fp:
                w=csv.DictWriter(fp,fieldnames=['model_id','sample_id','feature','class_name','shap_value']);w.writeheader()
                for mid in models:
                    for sample in samples:
                        for fidx,feat in enumerate(features):
                            for cidx,cls in enumerate(classes):
                                margin=1 if (mid=='a' or sample==0) else (-1 if mid=='b' else 1)
                                w.writerow(dict(model_id=mid,sample_id=sample,feature=feat,class_name=cls,
                                    shap_value=(margin if fidx==0 and cidx==1 else 0)))
            p=[]
            for mid in models:
                p.extend([{'model_id':mid,'sample_id':0,'y_pred':1},
                          {'model_id':mid,'sample_id':1,'y_pred':int(mid!='b')}])
            pd.DataFrame(p).to_csv(root/'test_predictions.csv',index=False)
            result=analyze(root,root/'out',(0.1,))
            row=result.iloc[0]
            self.assertAlmostEqual(row.prediction_same_explanation_same,0.5)
            self.assertAlmostEqual(row.prediction_different_explanation_different,0.5)
            self.assertEqual(len(pd.read_csv(root/'out'/'quadrant_sample_metrics.csv')),2)

if __name__=='__main__':unittest.main()
