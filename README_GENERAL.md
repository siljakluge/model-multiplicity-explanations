# Model Multiplicity and Explanations — general tabular benchmarks

Factorized experiments for **prediction multiplicity** and **explanation multiplicity** in *binary tabular classification*. This checkout runs general-ML benchmarks; NSL-KDD is deliberately excluded from the active defaults. Historical data in `paper/` are retained for provenance only and must not be included in general-ML aggregations.

## Datasets

| Full config | `dataset.name` | Origin / label | N (approx.) |
|---|---|---|---:|
| `configs/adult_full.yaml` | `adult` | UCI Adult train/test | 48,842 |
| `configs/folktables_income_full.yaml` | `folktables`, `problem: ACSIncome` | 2018 ACS CA, person | variable |
| `configs/folktables_employment_full.yaml` | `folktables`, `problem: ACSEmployment` | 2018 ACS CA, person | variable |
| `configs/bank_marketing_full.yaml` | `bank_marketing` | UCI bank-full (retrospective, retains call `duration`) | 45,211 |
| `configs/compas_full.yaml` | `compas` | ProPublica COMPAS two-year cohort | ~6,000 |
| `configs/german_credit_full.yaml` | `german_credit` | UCI Statlog German Credit | 1,000 |
| `configs/taiwan_credit_full.yaml` | `taiwan_credit` | UCI Default of Credit Card Clients | 30,000 |
| `configs/breast_cancer_full.yaml` | `breast_cancer` | `sklearn.datasets.load_breast_cancer` | 569 |

Data download automatically respects system HTTP(S)_PROXY. Cached files are stored in the configured `dataset.raw_dir`. Taiwan credit requires `xlrd>=2` to read the official `.xls` file.

**Study limitations:** COMPAS is a sensitive criminal-justice dataset; race and sex are features by default. Document cohort filtering and assess subgroups. German Credit/Breast Cancer have small test sets and are supporting robustness benchmarks. ACSIncome and Adult are related tasks rather than independent domains. Bank Marketing contains post-call `duration`, suitable only for retrospective analysis; a deployment study would need a no-duration ablation. Multiclass tasks are not part of this benchmark set.

## Quick start

```bash
pip install -e .
python -m unittest discover -s tests -v
python -m mmcyber.cli preflight
```

Run **one** new dataset to validate the pipeline:

```bash
python -m mmcyber.cli run-all --config configs/compas_full.yaml \
  --origin-seeds 1 --rashomon-tolerance 0.015 \
  --max-background 32 --max-explain 64 --top-k 5 \
  --aggregate-out-dir runs/aggregate_compas_pilot
```

Run all eight datasets (`10 x 63` models per dataset; resource intensive):

```bash
bash scripts/run_all_full_datasets.sh
```

Output directories use distinct dataset names and origin seeds. `preflight` checks plans and downloads without training. Use a **fresh output directory** when changing dataset or training parameters.

## Four-case prediction/explanation analysis

After `run-all`, recompute sample-level quadrants using decision-margin SHAP (no training):

```bash
python analysis/prediction_explanation_quadrants.py \
  --run-dir runs/compas_multiplicity_seed1 \
  --thresholds 0.05 0.10 0.15
```

The script emits `quadrant_sample_metrics.csv` and `quadrant_threshold_sensitivity.csv` in the run's `quadrant_analysis/` directory. These tables refer to the **SHAP-explained sample only** (default 256), not all test samples. The script uses all selected models from `shap_manifest.json`; their selected Rashomon set and test predictions must match.

## Paper-methodology notes

- All models are MLP variants (seeds, resampling, architecture/activation, and subset sizes). The factorized bootstrap block also varies model initialization, so its effect is not an isolated causal bootstrap effect.
- The selection rule uses validation accuracy within an absolute tolerance of the best validation accuracy. Report sensitivity for `epsilon in {0.001, 0.002, 0.005, 0.01, 0.015}`; avoid implying the full ensemble is always an equally strong Rashomon set.
- `mmcyber/shap_worker.py` now records the **full feature ranking**, preventing truncated-importance cosine distances. `explanation_disagreement.py` now defaults to `max(3, ceil(10% of transformed features))`, capped at `d-1`, and records the effective k. Report the *effective* k.
- Existing SHAP outputs from an older code revision must be regenerated for full feature rankings. Do not combine old and new runs without checking provenance.
- For the primary paper analysis, explain the **binary decision margin** `logit_1 - logit_0`, rather than averaging explanations of two independent logits. The original SHAP worker still emits class-specific logits; compute the margin by subtracting their attribution vectors before your quadrant analysis.
- For the four-case table, clearly distinguish unanimous ensemble agreement at a sample from pairwise agreement among model pairs; report multiple explanation similarity thresholds and continuous correlations.
- Since `train.py` uses `origin_disjoint`, test/validation splits vary between origins. Report uncertainty at the origin level, rather than treating correlated model pairs or SHAP samples as independent repetitions.

## Source URLs

- German Credit: https://archive.ics.uci.edu/dataset/144/statlog-german-credit-data
- COMPAS: https://github.com/propublica/compas-analysis (ProPublica cohort, `compas-scores-two-years.csv`)
- Taiwan: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Breast Cancer: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Selective plots and multiple explanation methods

See [GENERAL_ML_MIGRATION.md](GENERAL_ML_MIGRATION.md) for `none`, `rankings`, `paper`, `full` plot profiles, repaired BA Feature Ranking plots, historical SHAP summary rebuilding, and optional **SHAP margin / Captum Integrated Gradients / grouped-feature LIME** comparisons. Install the optional methods with `pip install -e '.[explain]'`.
