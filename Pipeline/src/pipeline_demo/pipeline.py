from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    # Python 3.9+
    from importlib.resources import files as ir_files
except Exception:  # pragma: no cover
    # Python <=3.8
    import importlib.resources as ir
    ir_files = lambda pkg: ir.files(pkg)  # type: ignore


NUMERIC_FEATURES = ["age", "income_gtq", "monthly_visits"]
CATEGORICAL_FEATURES = ["city", "has_kids"]
TARGET = "churn"
DATA_PKG = "pipeline_demo.data"
DATA_FILENAME = "synthetic_customers.csv"


def _default_data_path() -> str:
    """Return absolute path to the packaged CSV data."""
    try:
        path = ir_files(DATA_PKG).joinpath(DATA_FILENAME)
        return str(path)
    except Exception:
        # Fallback to package-relative path if importlib.resources is limited
        return os.path.join(os.path.dirname(__file__), "data", DATA_FILENAME)


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the demo dataset from CSV.

    If path is None, load the packaged CSV bundled with the distribution.
    """
    csv_path = path or _default_data_path()
    return pd.read_csv(csv_path)


def build_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer for numeric and categorical preprocessing."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_pipeline() -> Pipeline:
    """Create the full sklearn Pipeline with preprocessing and classifier."""
    pre = build_preprocessor()
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", clf),
    ])
    return pipe


@dataclass
class TrainResult:
    pipeline: Pipeline
    accuracy: float
    report: str


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainResult:
    """Train the pipeline and evaluate on a held-out test set.

    Returns a TrainResult with the fitted pipeline and metrics.
    """
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0)
    return TrainResult(pipeline=pipe, accuracy=acc, report=report)


def cli_train(argv: Optional[list[str]] = None) -> int:
    """Console entry point: train on the packaged data and print metrics.

    Usage: pipeline-demo-train [--data PATH] [--test-size 0.25] [--random-state 42]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train demo pipeline.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")

    args = parser.parse_args(argv)

    df = load_data(args.data)
    result = train_and_evaluate(df, test_size=args.test_size, random_state=args.random_state)

    print(f"Accuracy: {result.accuracy:.3f}")
    print(result.report)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_train())
