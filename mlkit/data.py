"""Data loading, validation, splitting, and scaling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    n_samples: int
    n_features: int
    n_classes: int


def load_dataset(
    path: str | Path,
    target_column: str,
    test_size: float = 0.2,
    scale: bool = True,
    random_state: int = 42,
    drop_columns: Optional[list[str]] = None,
) -> DataSplit:
    """Load a CSV file and return a ready-to-train DataSplit.

    Parameters
    ----------
    path            Path to a CSV file.
    target_column   Name of the column to predict.
    test_size       Fraction of data to hold out for testing (default 0.2).
    scale           Whether to apply StandardScaler to numeric features.
    random_state    Random seed for reproducibility.
    drop_columns    Extra columns to discard before training.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    # Drop rows with any NaN
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  [data] Dropped {dropped} row(s) containing NaN values.")

    y_raw = df[target_column].values
    X_df = df.drop(columns=[target_column])

    # Encode categorical features
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    # Encode target if it is categorical
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        n_samples=len(df),
        n_features=len(feature_names),
        n_classes=len(class_names),
    )


def describe_dataset(split: DataSplit) -> str:
    lines = [
        f"Samples   : {split.n_samples}",
        f"Features  : {split.n_features}  ({', '.join(split.feature_names[:5])}"
        + (" ..." if split.n_features > 5 else "") + ")",
        f"Classes   : {split.n_classes}  ({', '.join(split.class_names)})",
        f"Train/Test: {len(split.X_train)} / {len(split.X_test)}",
    ]
    return "\n".join(lines)
