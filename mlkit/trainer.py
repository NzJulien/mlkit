"""Train classifiers, cross-validate, and compute evaluation metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import DataSplit


@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    train_time_s: float
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    confusion: Optional[np.ndarray] = None
    report: str = ""
    fitted_model: object = field(default=None, repr=False)

    def summary_row(self) -> dict:
        row = {
            "Model": self.model_name,
            "Accuracy": f"{self.accuracy:.4f}",
            "Precision": f"{self.precision:.4f}",
            "Recall": f"{self.recall:.4f}",
            "F1": f"{self.f1:.4f}",
            "ROC-AUC": f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A",
            "Train (s)": f"{self.train_time_s:.2f}",
        }
        if self.cv_mean is not None:
            row["CV Mean"] = f"{self.cv_mean:.4f} ± {self.cv_std:.4f}"
        return row


def train(model, split: DataSplit) -> ModelResult:
    """Fit a classifier on the training split and evaluate on the test split."""
    display_name = type(model).__name__

    t0 = time.perf_counter()
    model.fit(split.X_train, split.y_train)
    train_time = time.perf_counter() - t0

    return evaluate(model, split, display_name, train_time)


def evaluate(model, split: DataSplit, name: Optional[str] = None, train_time: float = 0.0) -> ModelResult:
    """Evaluate a fitted model against the test split."""
    name = name or type(model).__name__
    y_pred = model.predict(split.X_test)

    avg = "binary" if split.n_classes == 2 else "weighted"

    accuracy  = accuracy_score(split.y_test, y_pred)
    precision = precision_score(split.y_test, y_pred, average=avg, zero_division=0)
    recall    = recall_score(split.y_test, y_pred, average=avg, zero_division=0)
    f1        = f1_score(split.y_test, y_pred, average=avg, zero_division=0)
    confusion = confusion_matrix(split.y_test, y_pred)
    report    = classification_report(split.y_test, y_pred, target_names=split.class_names, zero_division=0)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(split.X_test)
            if split.n_classes == 2:
                roc_auc = roc_auc_score(split.y_test, proba[:, 1])
            else:
                roc_auc = roc_auc_score(split.y_test, proba, multi_class="ovr", average="weighted")
        except Exception:
            pass

    return ModelResult(
        model_name=name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        train_time_s=train_time,
        confusion=confusion,
        report=report,
        fitted_model=model,
    )


def cross_validate_model(model, split: DataSplit, cv: int = 5) -> tuple[float, float]:
    """Run stratified k-fold cross-validation on the full data and return (mean, std)."""
    import numpy as np
    X = np.vstack([split.X_train, split.X_test])
    y = np.concatenate([split.y_train, split.y_test])
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return float(scores.mean()), float(scores.std())
