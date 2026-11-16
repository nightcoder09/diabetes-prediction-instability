# Equally Accurate, Not Equally Reliable

Code for the study **"Equally Accurate, Not Equally Reliable: Measuring and Reducing Individual
Prediction Instability in Ensemble Diabetes Risk Models"**.

Models that are equally accurate overall can still give the same person different answers when
they are re-trained. This code measures how often that happens for diabetes risk models built on
the CDC's Behavioral Risk Factor Surveillance System (BRFSS) survey, explains where the
instability comes from, and tests a simple way to reduce it. It contains:

- a leakage-safe pipeline: data download and validation, an 80/20 split, tuning and training on
  development data only, and a single evaluation on the held-out test set;
- a stacking ensemble of logistic regression, XGBoost, LightGBM and CatBoost in log-odds space,
  with exact SHAP explanations;
- a reliability audit: calibration, individual prediction instability across 20 re-trainings,
  explanation stability, robustness, subgroup equity and selective prediction (deferral);
- temporal validation on BRFSS 2021 and 2023, rebuilt from the raw CDC files;
- a seeds-versus-samples analysis and a stability-aware model-selection remedy.

> **Research use only.** The models are trained on self-reported, cross-sectional survey data. Their
> outputs are not validated estimates of an individual's clinical risk and must not guide individual care.

---

## 1. Quick start (macOS / Linux)

Requirements: Python 3.10-3.13, about 2 GB free disk, and internet for the first data download.
CPU only; no GPU is needed.

```bash
# macOS only: the OpenMP runtime used by XGBoost and LightGBM
brew install libomp

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

pytest                                                   # tests on synthetic data (~3-5 min)
python scripts/run_all.py --config configs/smoke.yaml    # fast end-to-end check; results NOT valid
python scripts/run_all.py --config configs/default.yaml  # the full study (steps 01-13)
```

`make setup`, `make test`, `make smoke`, `make all` and `make predict` are shortcuts. In VS Code,
`.vscode/launch.json` provides ready-made run configurations.

Measured run time for steps 01-13 with `configs/default.yaml`: about 3.7 h on a 2-core Linux machine
and about 1.4 h on a 14-core Apple-silicon Mac.

## 2. Data

No data are stored in this repository; the scripts download them.

- **BRFSS 2015 extract.** `configs/default.yaml` downloads the CDC Diabetes Health Indicators
  dataset from the UCI Machine Learning Repository (id 891, doi:10.24432/C53919). The same data are
  on Kaggle (<https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset>):
  put `diabetes_binary_health_indicators_BRFSS2015.csv` and
  `diabetes_012_health_indicators_BRFSS2015.csv` in `data/raw/kaggle/` and use
  `configs/kaggle_local.yaml`. Both sources give identical splits; `data/processed/metadata.json`
  records a SHA-256 content hash so you can confirm which data you analysed.
- **Raw BRFSS annual files** (for the temporal validation, about 0.6 GB):

```bash
mkdir -p data/raw/brfss && cd data/raw/brfss
for y in 2015 2021 2023; do curl -LO https://www.cdc.gov/brfss/annual_data/$y/files/LLCP${y}XPT.zip; done
cd ../../..
```

## 3. Pipeline

| Step | Script | What it does | Reads test labels? |
|---|---|---|---|
| 01 | `01_get_data.py` | download, schema validation, canonical ordering, provenance | - |
| 02 | `02_eda.py` | dataset summary, variable table, prevalence figure | - |
| 03 | `03_make_splits.py` | stratified 80/20 test split, 5 stratified development folds | - |
| 04 | `04_tune.py` | random search, 3-fold AUROC, development data only | no |
| 05 | `05_train.py` | out-of-fold predictions, stacking, calibration, thresholds, then **freeze** | no |
| 06 | `06_evaluate_test.py` | single test evaluation, bootstrap CIs, DeLong tests, decision curves | **yes (first time)** |
| 07 | `07_explain.py` | exact SHAP, cross-model agreement, perturbation sensitivity | yes |
| 08 | `08_multiplicity.py` | 20 full re-trainings: individual prediction instability, explanation stability | yes |
| 09 | `09_robustness.py` | missing-value and noise stress tests | yes |
| 10 | `10_fairness.py` | subgroup metrics with bootstrap CIs | yes |
| 11 | `11_selective.py` | deferral and risk-coverage analysis | yes |
| 12 | `12_ablations.py` | weighting, member removal, calibration, bagging, prediabetes | yes |
| 13 | `13_make_report.py` | all tables (CSV, Markdown, LaTeX) and figures | - |
| 08b | `08_multiplicity.py --seed-only` | 5 re-trainings on the full development set where only the seeds change | yes |
| 14 | `14_build_external.py` | rebuild BRFSS 2015/2021/2023 from the raw CDC files; checks 2015 against the published extract | no |
| 15 | `15_external_validation.py` | frozen models on 2021 and 2023: recalibration, drift, subgroups, SHAP drift | later years |
| 16 | `16_compare_runs.py` | compare two complete runs (for example on two computers) metric by metric | - |
| 17 | `17_ieee_figures.py` | compact two-column figures from saved results | - |
| 18 | `18_remedy.py` | stability-aware choice among equally accurate settings plus seed averaging | yes (one read of the frozen stable stack) |
| - | `csv_to_numpy.py`, `show_numpy.py` | the data as NumPy arrays with the study's split (`data/numpy/`) | - |

Every step can be resumed: `python scripts/run_all.py --from-step 8`. No step after 05 changes a
model, weight, calibrator or threshold. Outputs go to `experiments/`, `results/` and `figures/`.

Steps 14-18 run after the main analysis:

```bash
python scripts/14_build_external.py --config configs/default.yaml
python scripts/15_external_validation.py --config configs/default.yaml
python scripts/08_multiplicity.py --config configs/default.yaml            # adds later-year predictions
python scripts/08_multiplicity.py --config configs/default.yaml --seed-only
python scripts/15_external_validation.py --config configs/default.yaml --sensitivity-only
python scripts/13_make_report.py --config configs/default.yaml
python scripts/18_remedy.py --config configs/default.yaml --step all      # about 3 h on 2 cores
```

## 4. Method in one paragraph

Six baselines (flexible logistic regression with splines, decision tree, random forest, XGBoost,
LightGBM, CatBoost) are tuned on the development set. The proposed model stacks four members with an
additive log-odds output, `g(x) = b + Σ_k w_k m_k(x)` with `w_k ≥ 0`, and the weights are fitted by
penalised maximum likelihood on out-of-fold margins only. Because `g` is linear in the members, the
ensemble's SHAP values are the weighted sum of the members' SHAP values: exact, with local accuracy
checked numerically. Single models get one Platt layer fitted on out-of-fold predictions; the
screening threshold (80% sensitivity) is fixed on development data. The frozen models are then audited
on the held-out test set.

## 5. Predictions for new records

After steps 01-05 have been run:

```bash
python scripts/predict.py --input data/example_input.csv --output results/example_predictions.csv
python scripts/predict.py --input data/example_input.csv --stable    # the stable stack from step 18
```

Inputs must use the BRFSS coding (written to `results/dataset_variables.csv` by step 02). Out-of-range
or missing values are rejected with a clear message. The output gives the model score, the
screen-positive flag at the frozen threshold, member disagreement and the top SHAP contributions.

## 6. Reproducibility

- One master seed (`project.seed` in the config); every other seed is derived from it.
- Exact package versions: `requirements-lock.txt`; environment details are written to
  `results/environment.json`.
- Tree-boosting libraries use platform-specific floating-point code. In a re-run on a second computer
  every test metric agreed within 0.00012 AUROC, but two XGBoost tuning candidates that were tied to
  0.00004 cross-validated AUROC swapped order. `scripts/16_compare_runs.py` reports such differences.

## 7. Known limitations

- The re-training analysis covers the four stacking members and a bagged CatBoost; random forest and
  decision tree are not re-trained there (cost).
- Temporal validation stays within one survey system (BRFSS 2021, 2023). The fruit and vegetable items
  were not asked in 2023 and are imputed with development medians.
- Robustness tests use one rule for missing values (development-set medians) for every model.

## 8. Layout

```
configs/    default.yaml (main study), kaggle_local.yaml, repro_mac.yaml, smoke*.yaml
src/ehai/   data, models, ensemble, calibration, metrics, stats, tuning, pipeline, experiments,
            explain, robustness, fairness, selective, remedy, external, brfss_raw, predict, plotting
scripts/    pipeline steps 01-18, run_all.py, predict.py, csv_to_numpy.py, show_numpy.py
tests/      unit tests and an end-to-end integration test (synthetic data only)
notebooks/  open_numpy_data.ipynb
data/       example_input.csv (everything else is downloaded)
```

## 9. Licence and citation

Code: MIT licence (see `LICENSE`). Data: CDC Diabetes Health Indicators, UCI Machine Learning
Repository, doi:10.24432/C53919; raw BRFSS annual files from the CDC
(<https://www.cdc.gov/brfss/annual_data/annual_2015.html> and the 2021 and 2023 equivalents).

If you use this code, please cite:
R. Kumar and S. Aswath, "Equally Accurate, Not Equally Reliable: Measuring and Reducing Individual
Prediction Instability in Ensemble Diabetes Risk Models," manuscript in preparation, 2026.
