import pytest
import pandas as pd
from mlkit.data import load_dataset, describe_dataset


def make_csv(tmp_path, rows=100, classes=2):
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "feat_a": rng.standard_normal(rows),
        "feat_b": rng.standard_normal(rows),
        "feat_c": rng.integers(0, 10, rows).astype(float),
        "label": [f"class_{i % classes}" for i in range(rows)],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_load_dataset_basic(tmp_path):
    path = make_csv(tmp_path)
    split = load_dataset(path, "label")
    assert split.n_samples == 100
    assert split.n_features == 3
    assert split.n_classes == 2
    assert len(split.X_train) == 80
    assert len(split.X_test) == 20


def test_load_dataset_feature_names(tmp_path):
    path = make_csv(tmp_path)
    split = load_dataset(path, "label")
    assert split.feature_names == ["feat_a", "feat_b", "feat_c"]


def test_load_dataset_class_names(tmp_path):
    path = make_csv(tmp_path, classes=3)
    split = load_dataset(path, "label", test_size=0.25)
    assert set(split.class_names) == {"class_0", "class_1", "class_2"}


def test_load_dataset_scales_by_default(tmp_path):
    import numpy as np
    path = make_csv(tmp_path)
    split = load_dataset(path, "label", scale=True)
    # Scaled training features should have roughly zero mean and unit variance
    assert abs(split.X_train.mean()) < 0.1
    assert abs(split.X_train.std() - 1.0) < 0.1


def test_load_dataset_no_scale(tmp_path):
    import numpy as np
    path = make_csv(tmp_path)
    split_scaled   = load_dataset(path, "label", scale=True)
    split_unscaled = load_dataset(path, "label", scale=False)
    # Unscaled std should be larger (original feature scale)
    assert split_unscaled.X_train.std() != pytest.approx(1.0, abs=0.1)


def test_load_dataset_missing_target_raises(tmp_path):
    path = make_csv(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        load_dataset(path, "nonexistent_column")


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "ghost.csv", "label")


def test_load_dataset_drops_nan_rows(tmp_path):
    import numpy as np
    path = make_csv(tmp_path, rows=50)
    df = pd.read_csv(path)
    df.loc[0, "feat_a"] = float("nan")
    df.to_csv(path, index=False)
    split = load_dataset(path, "label")
    assert split.n_samples == 49


def test_describe_dataset_string(tmp_path):
    path = make_csv(tmp_path)
    split = load_dataset(path, "label")
    desc = describe_dataset(split)
    assert "Samples" in desc
    assert "Features" in desc
    assert "Classes" in desc
