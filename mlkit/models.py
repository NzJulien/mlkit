"""Registry of supported classifiers with sensible defaults."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Registry maps short name -> (display_name, factory_function)
MODELS: dict[str, tuple[str, callable]] = {
    "lr": (
        "Logistic Regression",
        lambda: LogisticRegression(max_iter=1000, random_state=42),
    ),
    "dt": (
        "Decision Tree",
        lambda: DecisionTreeClassifier(random_state=42),
    ),
    "rf": (
        "Random Forest",
        lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    ),
    "gb": (
        "Gradient Boosting",
        lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    ),
    "svm": (
        "Support Vector Machine",
        lambda: SVC(probability=True, random_state=42),
    ),
    "knn": (
        "K-Nearest Neighbors",
        lambda: KNeighborsClassifier(n_neighbors=5),
    ),
    "nb": (
        "Naive Bayes",
        lambda: GaussianNB(),
    ),
}


def get_model(name: str):
    """Return a fresh (unfitted) classifier instance by short name."""
    name = name.lower().strip()
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(MODELS)}"
        )
    _, factory = MODELS[name]
    return factory()


def list_models() -> list[tuple[str, str]]:
    """Return [(short_name, display_name), ...] sorted by short_name."""
    return [(k, v[0]) for k, v in sorted(MODELS.items())]
