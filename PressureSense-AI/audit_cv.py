from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


def main() -> None:
    parser = argparse.ArgumentParser(description="Animal-grouped overfitting audit")
    parser.add_argument("--features", default="data/processed/training_features.npz")
    parser.add_argument("--model", default="models/goat_motion.joblib")
    parser.add_argument("--profile", default="cabritrack")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="reports/cross_validation.json")
    args = parser.parse_args()

    cached = np.load(args.features)
    X, y, groups = cached["X"], cached["y"], cached["groups"].astype(str)
    mask = np.char.startswith(groups, f"{args.profile}:")
    X, y, groups = X[mask], y[mask], groups[mask]
    bundle = joblib.load(args.model)
    template = bundle["behavior_classifiers"][args.profile]
    labels = sorted(np.unique(y).tolist())
    folds = []
    splitter = GroupKFold(n_splits=args.folds)
    for fold_number, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
        model = clone(template)
        model.fit(X[train], y[train])
        train_prediction = model.predict(X[train])
        test_prediction = model.predict(X[test])
        folds.append({
            "fold": fold_number,
            "train_animals": int(len(np.unique(groups[train]))),
            "test_animals": int(len(np.unique(groups[test]))),
            "train_accuracy": float(accuracy_score(y[train], train_prediction)),
            "test_accuracy": float(accuracy_score(y[test], test_prediction)),
            "test_f1": {
                label: float(value) for label, value in zip(
                    labels,
                    f1_score(y[test], test_prediction, labels=labels, average=None, zero_division=0),
                )
            },
        })
    result = {
        "profile": args.profile,
        "method": f"{args.folds}-fold GroupKFold by animal",
        "folds": folds,
        "mean_train_accuracy": float(np.mean([fold["train_accuracy"] for fold in folds])),
        "mean_test_accuracy": float(np.mean([fold["test_accuracy"] for fold in folds])),
        "mean_generalization_gap": float(
            np.mean([fold["train_accuracy"] for fold in folds])
            - np.mean([fold["test_accuracy"] for fold in folds])
        ),
        "std_test_accuracy": float(np.std([fold["test_accuracy"] for fold in folds])),
        "mean_test_f1": {
            label: float(np.mean([fold["test_f1"][label] for fold in folds])) for label in labels
        },
        "std_test_f1": {
            label: float(np.std([fold["test_f1"][label] for fold in folds])) for label in labels
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
