import numpy as np
import pandas as pd
import pytest

from mlkit.data import load_dataset
from mlkit.models import MODELS, get_model, list_models
from mlkit.trainer import train, evaluate, cross_validate_model


# ---- Helpers ----

def make_split(tmp_path, rows=120, classes=2):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x1": rng.standard_normal(rows),
        "x2": rng.standard_normal(rows),
        "label": [i % classes for i in range(rows)],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, "label")


# ---- Model registry ----

def test_all_model_keys_present():
    for key in ["lr", "dt", "rf", "gb", "svm", "knn", "nb"]:
        assert key in MODELS


def test_get_model_returns_fresh_instance():
    m1 = get_model("lr")
    m2 = get_model("lr")
    assert m1 is not m2


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("zzz")


def test_list_models_sorted():
    names = [k for k, _ in list_models()]
    assert names == sorted(names)


# ---- Trainer ----

def test_train_returns_model_result(tmp_path):
    split = make_split(tmp_path)
    model = get_model("lr")
    result = train(model, split)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.f1 >= 0.0
    assert result.train_time_s >= 0.0
    assert result.confusion is not None
    assert result.report != ""


def test_train_all_models(tmp_path):
    split = make_split(tmp_path)
    for key in MODELS:
        model = get_model(key)
        result = train(model, split)
        assert result.accuracy >= 0.0, f"{key} returned invalid accuracy"


def test_train_multiclass(tmp_path):
    split = make_split(tmp_path, classes=3)
    model = get_model("dt")
    result = train(model, split)
    assert split.n_classes == 3
    assert result.f1 >= 0.0


def test_cross_validate_returns_mean_and_std(tmp_path):
    split = make_split(tmp_path)
    model = get_model("lr")
    mean, std = cross_validate_model(model, split, cv=3)
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0


def test_roc_auc_computed_for_binary(tmp_path):
    split = make_split(tmp_path)
    model = get_model("lr")
    result = train(model, split)
    assert result.roc_auc is not None
    assert 0.0 <= result.roc_auc <= 1.0


def test_summary_row_keys(tmp_path):
    split = make_split(tmp_path)
    result = train(get_model("lr"), split)
    row = result.summary_row()
    for key in ["Model", "Accuracy", "F1"]:
        assert key in row
