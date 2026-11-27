# All-in-one migration for `siljakluge/model-multiplicity-explanations`

This bundle targets the `main` repository structure inspected **2026-11-27**.

## Apply once to a clean checkout

```bash
git clone https://github.com/siljakluge/model-multiplicity-explanations.git
cd model-multiplicity-explanations
git switch -c feature/general-multiplicity-paper
# Extract model_multiplicity_complete.zip here, yielding model_multiplicity_complete/
python model_multiplicity_complete/apply.py .
python -m pip install -e '.[explain]'
python -m unittest discover -s tests -v
python -m mmcyber.cli preflight
```

The installer **does not push to GitHub, delete archived NSL-KDD results or delete runs**. It backs up modified files under `docs/migration_backups/complete_2026-11-27/` and refuses unrecognized earlier partial patches. Use a fresh main checkout if you previously applied the standalone packages; merge user changes afterwards using `git diff`.

The preflight can download data from the official sources; on restricted servers, supply the required HTTPS proxy or place official raw files in the specified cache locations. On the company server, check data licenses and internal policies before transferring downloaded datasets.

## Datasets

The four existing datasets (Adult, ACSIncome, ACSEmployment, Bank Marketing) remain. Four additional datasets are provided: ProPublica COMPAS, Statlog German Credit, Taiwan Credit Default, Wisconsin Breast Cancer. Historical NSL-KDD artifacts under `paper/` are left untouched but NSL-KDD is absent from the active `default` and the full runner. The default config is copied from the existing Folktables ACSIncome full config; `configs/smoke.yaml` from its existing smoke variant.

New datasets: `configs/{compas,german_credit,taiwan_credit,breast_cancer}_full.yaml`.

**Do not treat the datasets as identical tasks:** COMPAS has recidivism outcome/provenance limitations, Breast Cancer is small, Adult and ACSIncome overlap semantically, and Bank Marketing includes post-call `duration` in the original experiment (retrospective analysis only). If removing duration, create a separate dataset configuration and fresh run directory. Training/validation/test preprocessing remains train-only.

## Preprocessing, SHAP and top-k

- Each newly loaded dataset provides `PreparedData` with stratified train/validation/test and no target/ID leakage.
- Original `shap_worker.py` now stores all feature ranks in `shap_summary.csv`, rather than truncating at 50.
- To repair **old** runs without rerunning the explainer (if NPZ caches exist):

  ```bash
  python analysis/rebuild_shap_summary.py --run-dir runs/adult_multiplicity_seed1
  ```

  This backs up the old CSV. If the original `shap_results` cache is later replayed, re-run the rebuild script.
- Explanation top-k uses relative feature count by default:

  `k = min(d - 1, max(3, ceil(0.10*d)))` for `d >= 2`, with `d` the *post-preprocessing* feature count.
- You can override it with `--top-k 5` or choose `--top-k-fraction 0.05`. Plotting has an **independent** `--plot-top-n` / `--top-n`.
- For one-hot-heavy datasets, interpret this transformed-feature ranking carefully. The new SHAP/IG/LIME method comparison also works on **grouped original features** for fairness.

## Plot modes

The updated CLI supports:

| Mode | Intended use |
|---|---|
| `none` | Default for `run-all`; no per-origin plots |
| `rankings` | BA Feature Ranking per class only, PNG and PDF |
| `paper` | Four short diagnostics plus both BA rank plots |
| `full` | Original exhaustive per-feature plots |

`Adult`'s `<=50K` and `>50K` labels are disambiguated in file names; labels `negative` and `positive` preserve the `ba_feature_ranking_negative/positive` names.

```bash
python -m mmcyber.cli run-all --config configs/compas_full.yaml \
  --origin-seeds 1 --rashomon-tolerance 0.015 \
  --max-background 32 --max-explain 64 \
  --top-k-fraction 0.10 --plot-profile none \
  --aggregate-out-dir runs/aggregate_compas_pilot

python -m mmcyber.cli plot --run-dir runs/compas_multiplicity_seed1 \
  --profile rankings --top-n 8
```

The numerical comparisons and attribution files are generated independently of plotting. A `plot --profile none` does not purge previous plots. To avoid old clutter, archive old `plots/` manually before regenerating.

## Explanation methods (optional)

### Underlying scientific target

All three methods explain the **binary decision margin** `z_1(x)-z_0(x)`:

- **SHAP:** difference of cached DeepSHAP class-logit attributions (not reexplained).
- **IG:** Captum Integrated Gradients with common training-mean baseline. Logs per-instance absolute completeness delta.
- **LIME:** regression-mode local ridge surrogate for the margin. It perturbs original categorical variables, not independent one-hot dummy columns. Logs local surrogate R². Its local coefficients are not identical estimands to SHAP/IG contributions.

To compare methods, SHAP and IG dummy-variable attributions are summed into original categorical feature groups. The same model IDs, sample IDs and reference input target are used across methods. `method_comparison/manifest.json` records experiment settings. This is an **additional optional analysis**; it does not silently replace the existing per-logit SHAP files, original experiment metrics or BA plots.

Start small:

```bash
python -m mmcyber.cli multi-explain \
  --run-dir runs/compas_multiplicity_seed1 \
  --methods shap ig lime --max-models 5 --max-samples 16 \
  --ig-steps 64 --lime-samples 1000
python analysis/compare_methods.py --run-dir runs/compas_multiplicity_seed1
```

For a full within-run comparison, omit `--max-models` and increase `--max-samples` to `256`; this can be expensive, especially for LIME. Use `--overwrite` when changing settings for an already-existing method comparison output. For studies of stochastic explainer variance, repeat LIME with multiple seeds and report within-model differences separately from between-model differences.

For old SHAP-only quadrants, continue using:

```bash
python analysis/prediction_explanation_quadrants.py \
  --run-dir runs/adult_multiplicity_seed1 --thresholds 0.05 0.10 0.15
```

### Publication caveats

- A 10% relative top-k is not the only principled choice. Supplement with fixed `k = 3,5` and `5%/20%` robustness checks.
- 0.10 cosine distance is an **exploratory threshold**, not a universal definition of explanation disagreement. Prefer continuous measurements as primary results.
- Compare methods using paired observations and sample- or origin-level uncertainty intervals, not model-pair-level pseudo-replication.
- LIME local coefficients and IG/SHAP attributions differ in semantics; cross-method rank or directional comparisons are descriptive unless a stronger equivalence is justified.
- DeepSHAP, IG and LIME results can depend on baselines, sampling and perturbations. Report method diagnostics, not only cross-model disagreement.
- If `--only-conflicts` was used for SHAP, any quadrant proportions are conditioned on conflict-only sampling and **must not** be presented as full-population proportions.

## Testing and limitations

The bundle's offline fixtures check new loaders, split and no-leakage behavior, grouped feature transformations, historical SHAP summary restoration and sample-level quadrant analysis. Tests do not verify external dataset availability or long GPU training. Run `preflight` and a pilot origin before a 5,040-model sweep.

Backups reside in `docs/migration_backups/complete_2026-11-27` and can be restored manually or reverted via git before merging.
