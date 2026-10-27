"""Data loading, validation, splitting, and leakage-safe preprocessing."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the raw Telco CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV {path}: {exc}") from exc
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate the required target and binary label contract."""
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' column.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    labels = set(frame[TARGET].dropna().astype(str).unique())
    if labels != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError("Churn must contain both 'No' and 'Yes' labels.")


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert known numeric fields without fitting anything on evaluation data."""
    prepared = frame.copy()
    if "TotalCharges" in prepared.columns:
        prepared["TotalCharges"] = pd.to_numeric(prepared["TotalCharges"], errors="coerce")
    return prepared


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return raw model features, excluding the identifier and target."""
    return [column for column in frame.columns if column not in {ID_COLUMN, TARGET}]


def split_dataset(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the one reproducible, stratified split used by every model."""
    prepared = prepare_frame(frame)
    features = prepared[feature_columns(prepared)]
    target = prepared[TARGET].map({NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1})
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_preprocessor(frame: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a transformer whose imputers/encoders are fitted inside each pipeline."""
    columns = feature_columns(frame)
    numeric = frame[columns].select_dtypes(include="number").columns.tolist()
    categorical = [column for column in columns if column not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def schema_metadata(frame: pd.DataFrame) -> Dict[str, object]:
    """Return portable schema information for app validation and reporting."""
    columns = feature_columns(frame)
    numeric = frame[columns].select_dtypes(include="number").columns.tolist()
    return {
        "input_columns": columns,
        "numeric_columns": numeric,
        "categorical_columns": [column for column in columns if column not in numeric],
        "target": TARGET,
        "target_mapping": {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1},
    }