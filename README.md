# mlkit

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen)

**Train and compare scikit-learn classifiers on any CSV dataset — one command.**

## Install

```bash
git clone https://github.com/NzJulien/mlkit.git
cd mlkit
pip install -e .
```

## Quick start

```bash
# Inspect your dataset
mlkit info data.csv --target label

# Train one model
mlkit train data.csv --target label --model rf

# Train all 7 models and compare
mlkit compare data.csv --target label --cv 5
```

```
  Model                    Accuracy  F1      ROC-AUC  Train (s)
  -----------------------  --------  ------  -------  ---------
  Random Forest            0.9650    0.9649  0.9941   0.23
  Gradient Boosting        0.9600    0.9599  0.9935   0.51
  Support Vector Machine   0.9550    0.9548  0.9928   0.04
  Logistic Regression      0.9200    0.9198  0.9810   0.02
  K-Nearest Neighbors      0.9150    0.9148  0.9750   0.01
  Decision Tree            0.8900    0.8897  0.8901   0.01
  Naive Bayes              0.8750    0.8748  0.9601   0.01

  Best model by F1: Random Forest  (F1=0.9649)
```

## Models

| Key  | Model |
|------|-------|
| `lr`  | Logistic Regression |
| `dt`  | Decision Tree |
| `rf`  | Random Forest |
| `gb`  | Gradient Boosting |
| `svm` | Support Vector Machine |
| `knn` | K-Nearest Neighbors |
| `nb`  | Naive Bayes |

## All commands

```bash
mlkit info data.csv --target label           # Dataset summary
mlkit models                                 # List model keys
mlkit train data.csv --target label --model rf svm lr   # Train specific models
mlkit train data.csv --target label --cv 5 -v           # CV + verbose report
mlkit compare data.csv --target label -o report.txt     # All models, save report
```

## Library usage

```python
from mlkit import load_dataset, get_model, train
from mlkit.compare import compare_table, best_model

split = load_dataset("data.csv", target_column="label")
results = [train(get_model(k), split) for k in ["lr", "rf", "svm"]]

print(compare_table(results))
print("Best:", best_model(results).model_name)
```

## Project layout

```
mlkit/
├── mlkit/
│   ├── data.py      # CSV loading, NaN dropping, scaling, label encoding
│   ├── models.py    # Registry of 7 classifiers
│   ├── trainer.py   # train(), evaluate(), cross_validate_model()
│   ├── compare.py   # comparison table, best_model(), full_report()
│   └── cli.py       # argparse CLI: info, models, train, compare
├── tests/           # pytest suite (19 tests)
├── requirements.txt
└── setup.py
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest -v
```

Made by [NzJulien](https://github.com/NzJulien)
