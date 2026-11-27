"""Additional binary tabular benchmarks for predictive/explanatory multiplicity.

The official data are cached under dataset.raw_dir. All columns are selected
explicitly to avoid outcome leakage. Train/validation/test splitting is
stratified and preprocessing is fitted only on the training partition.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, BadZipFile
import shutil

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mmcyber.data import PreparedData, _download, _fit_and_transform, _save_preprocessor_artifacts

GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
TAIWAN_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

GERMAN_FEATURES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker",
]
GERMAN_NUMERIC = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "dependents",
]
COMPAS_FEATURES = [
    "age", "sex", "race", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "c_charge_degree",
]
COMPAS_NUMERIC = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
TAIWAN_CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def _read_or_download_file(path: Path, url: str) -> Path:
    if not path.is_file():
        _download(url, path)  # existing repo's proxy-aware and atomic downloader
    if path.stat().st_size == 0:
        raise ValueError(f"Empty dataset cache: {path}")
    return path


def _prepare_binary_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    dataset_config: dict,
    run_dir: Path,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    class_names: list[str],
) -> PreparedData:
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Empty feature table or duplicated feature column names")
    if frame.isna().all(axis=0).any():
        raise ValueError("One or more input feature columns contain only missing values")
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if len(y) != len(frame) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected two nonempty binary classes and matching features")
    if set(numeric_cols + categorical_cols) != set(frame.columns):
        raise ValueError("Numeric and categorical feature specification does not cover X")
    if set(numeric_cols) & set(categorical_cols):
        raise ValueError("A feature cannot be both numeric and categorical")

    test_frac = float(dataset_config.get("test_size", 0.2))
    val_frac = float(dataset_config.get("validation_size", 0.2))
    if not 0 < test_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("test_size and validation_size must lie between 0 and 1")
    seed = int(dataset_config.get("split_seed", 42))
    x_trv, x_test, y_trv, y_test = train_test_split(
        frame, y, test_size=test_frac, random_state=seed, stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trv, y_trv, test_size=val_frac, random_state=seed, stratify=y_trv,
    )
    for label, ys in (("train", y_train), ("val", y_val), ("test", y_test)):
        if set(np.unique(ys)) != {0, 1}:
            raise ValueError(f"{label} partition lost one class; choose larger dataset or split")

    # Ensure missing categorical values remain a consistent object dtype.
    if categorical_cols:
        x_train, x_val, x_test = x_train.copy(), x_val.copy(), x_test.copy()
    for col in categorical_cols:
        for split in (x_train, x_val, x_test):
            split[col] = split[col].astype("string").astype(object).where(split[col].notna(), np.nan)
    x1, x2, x3, names, preprocessor = _fit_and_transform(
        x_train_df=x_train, x_val_df=x_val, x_test_df=x_test,
        numeric_cols=numeric_cols, categorical_cols=categorical_cols,
    )
    _save_preprocessor_artifacts(run_dir, preprocessor, names)
    return PreparedData(
        x_train=x1, x_val=x2, x_test=x3,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=names, class_names=class_names, preprocessor=preprocessor,
    )


def prepare_german_credit(config: dict, run_dir: Path) -> PreparedData:
    raw_dir = Path(config.get("raw_dir", "data/raw/german-credit"))
    path = _read_or_download_file(raw_dir / "german.data", GERMAN_URL)
    frame = pd.read_csv(path, sep=r"\s+", names=[*GERMAN_FEATURES, "class"], header=None)
    if config.get("validate_full_size", True) and frame.shape != (1000, 21):
        raise ValueError(f"Unexpected German Credit shape: {frame.shape}; expected (1000, 21)")
    if frame.shape[1] != 21 or set(frame["class"].dropna().unique()) != {1, 2}:
        raise ValueError("German Credit must have 20 attributes and labels 1=good, 2=bad")
    y = (frame.pop("class") == 2).to_numpy(dtype=np.int64)
    for column in GERMAN_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    cats = [c for c in GERMAN_FEATURES if c not in GERMAN_NUMERIC]
    return _prepare_binary_frame(
        frame, y, config, run_dir, numeric_cols=GERMAN_NUMERIC,
        categorical_cols=cats, class_names=["good", "bad"],
    )


def prepare_compas(config: dict, run_dir: Path) -> PreparedData:
    raw_dir = Path(config.get("raw_dir", "data/raw/compas"))
    path = _read_or_download_file(raw_dir / "compas-scores-two-years.csv", COMPAS_URL)
    frame = pd.read_csv(path, low_memory=False)
    required = set([*COMPAS_FEATURES, "two_year_recid", "days_b_screening_arrest", "is_recid", "score_text"])
    if required - set(frame.columns):
        raise ValueError(f"COMPAS missing fields: {sorted(required - set(frame.columns))}")
    if config.get("validate_full_size", True) and len(frame) < 7000:
        raise ValueError("COMPAS source appears incomplete")
    # The standard ProPublica cohort filters are applied prior to splitting.
    # COMPAS's own score is used ONLY for selecting the cohort, not as a predictor.
    days = pd.to_numeric(frame["days_b_screening_arrest"], errors="coerce")
    keep = (
        days.between(-30, 30)
        & frame["is_recid"].isin([0, 1])
        & frame["c_charge_degree"].isin(["F", "M"])
        & frame["score_text"].notna()
        & frame["score_text"].ne("N/A")
        & frame["two_year_recid"].isin([0, 1])
    )
    frame = frame.loc[keep].reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError("COMPAS filtering left too few observations")
    y = frame["two_year_recid"].to_numpy(dtype=np.int64)
    x = frame[COMPAS_FEATURES].copy()
    for col in COMPAS_NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return _prepare_binary_frame(
        x, y, config, run_dir, numeric_cols=COMPAS_NUMERIC,
        categorical_cols=[c for c in COMPAS_FEATURES if c not in COMPAS_NUMERIC],
        class_names=["no_recidivism", "recidivism"],
    )


def _taiwan_excel_file(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    xls_path = raw_dir / "default of credit card clients.xls"
    if xls_path.is_file():
        return xls_path
    zip_path = _read_or_download_file(raw_dir / "default_credit_clients.zip", TAIWAN_URL)
    try:
        with ZipFile(zip_path) as zf:
            members = [member for member in zf.namelist() if member.lower().endswith(".xls") and not member.startswith("__MACOSX")]
            if len(members) != 1:
                raise ValueError(f"Expected one .xls spreadsheet, found: {members}")
            with zf.open(members[0]) as src, xls_path.open("wb") as dest:
                shutil.copyfileobj(src, dest)
    except (BadZipFile, OSError, ValueError) as exc:
        xls_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not extract the official Taiwan Credit dataset from {zip_path}") from exc
    return xls_path


def prepare_taiwan_credit(config: dict, run_dir: Path) -> PreparedData:
    raw_dir = Path(config.get("raw_dir", "data/raw/taiwan-credit"))
    path = _taiwan_excel_file(raw_dir)
    try:
        # Official UCI XLS: first row contains generic X1..X23 names; second row
        # has descriptive attribute names and the actual data start afterwards.
        frame = pd.read_excel(path, header=1, engine="xlrd")
    except ImportError as exc:
        raise ImportError("Taiwan Credit requires xlrd>=2.0; pip install xlrd>=2.0") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    target_candidates = ["default payment next month", "default.payment.next.month", "Y"]
    targets = [col for col in target_candidates if col in frame.columns]
    if len(targets) != 1:
        raise ValueError(f"Could not identify Taiwan Credit outcome column. Columns: {list(frame.columns)}")
    target = targets[0]
    if config.get("validate_full_size", True) and len(frame) != 30000:
        raise ValueError(f"Expected 30000 Taiwan Credit rows, got {len(frame)}")
    if "ID" not in frame:
        raise ValueError("Taiwan Credit missing ID column; likely incorrect XLS header")
    y_numeric = pd.to_numeric(frame[target], errors="raise")
    if y_numeric.isna().any() or set(y_numeric.unique()) != {0, 1}:
        raise ValueError("Taiwan Credit has invalid binary outcome")
    y = y_numeric.to_numpy(dtype=np.int64)
    x = frame.drop(columns=[target, "ID"]).copy()
    if x.shape[1] != 23:
        raise ValueError(f"Expected 23 predictors for Taiwan Credit, got {x.shape[1]}")
    for column in x.columns:
        x[column] = pd.to_numeric(x[column], errors="raise")
    categorical_cols = [c for c in TAIWAN_CATEGORICAL if c in x.columns]
    if len(categorical_cols) != 3:
        raise ValueError("Taiwan Credit missing demographic categorical columns")
    return _prepare_binary_frame(
        x, y, config, run_dir,
        numeric_cols=[c for c in x.columns if c not in categorical_cols],
        categorical_cols=categorical_cols,
        class_names=["no_default", "default"],
    )


def prepare_breast_cancer(config: dict, run_dir: Path) -> PreparedData:
    # scikit-learn's bundled Wisconsin Diagnostic Breast Cancer dataset.
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data.copy()
    y = np.asarray(dataset.target, dtype=np.int64)
    return _prepare_binary_frame(
        x, y, config, run_dir, numeric_cols=x.columns.tolist(),
        categorical_cols=[], class_names=["malignant", "benign"],
    )


PREPARERS = {
    "german_credit": prepare_german_credit,
    "compas": prepare_compas,
    "taiwan_credit": prepare_taiwan_credit,
    "breast_cancer": prepare_breast_cancer,
}
