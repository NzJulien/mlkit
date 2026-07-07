"""mlkit — a scikit-learn ML toolkit.

Train, evaluate, and compare multiple classifiers on any CSV dataset
through a clean CLI and importable Python API.

Modules
-------
- data      : load, validate, split, and scale CSV datasets
- models    : registry of supported classifiers with sensible defaults
- trainer   : cross-validated training, grid search, evaluation metrics
- compare   : side-by-side model comparison table
- cli       : argparse CLI exposing every feature
"""
from .data import load_dataset, DataSplit
from .models import MODELS, get_model
from .trainer import train, evaluate, cross_validate_model, ModelResult

__version__ = "1.0.0"
__all__ = [
    "load_dataset", "DataSplit",
    "MODELS", "get_model",
    "train", "evaluate", "cross_validate_model", "ModelResult",
]
