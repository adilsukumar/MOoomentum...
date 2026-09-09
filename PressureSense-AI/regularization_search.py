"""Compare deliberately constrained classifiers using animal-grouped CV."""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


def candidates() -> dict:
    common = dict(
        n_estimators=250,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return {
        "rf_depth5_leaf40": RandomForestClassifier(max_depth=5, min_samples_leaf=40, **common),
        "rf_depth5_leaf50": RandomForestClassifier(max_depth=5, min_samples_leaf=50, **common),
        "rf_depth5_leaf60": RandomForestClassifier(max_depth=5, min_samples_leaf=60, **common),
        "rf_depth5_leaf80": RandomForestClassifier(max_depth=5, min_samples_leaf=80, **common),
    }


def main() -> None:
    cached = np.load("data/processed/training_features.npz")
    X, y, groups = cached["X"], cached["y"], cached["groups"].astype(str)
    mask = np.char.startswith(groups, "cabritrack:")
    X, y, groups = X[mask], y[mask], groups[mask]
    labels = sorted(np.unique(y).tolist())
    results = {}
    for name, template in candidates().items():
        folds = []
        for train, test in GroupKFold(n_splits=5).split(X, y, groups):
            template.fit(X[train], y[train])
            train_pred = template.predict(X[train])
            test_pred = template.predict(X[test])
            folds.append({
                "train_accuracy": accuracy_score(y[train], train_pred),
                "test_accuracy": accuracy_score(y[test], test_pred),
                "macro_f1": f1_score(y[test], test_pred, average="macro", zero_division=0),
                "class_f1": f1_score(
                    y[test], test_pred, labels=labels, average=None, zero_division=0
                ).tolist(),
            })
        train_accuracy = float(np.mean([f["train_accuracy"] for f in folds]))
        test_accuracy = float(np.mean([f["test_accuracy"] for f in folds]))
        results[name] = {
            "mean_train_accuracy": train_accuracy,
            "mean_test_accuracy": test_accuracy,
            "generalization_gap": train_accuracy - test_accuracy,
            "mean_macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
            "mean_class_f1": {
                label: float(np.mean([f["class_f1"][i] for f in folds]))
                for i, label in enumerate(labels)
            },
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
