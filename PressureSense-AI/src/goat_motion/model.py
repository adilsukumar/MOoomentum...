from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from .features import FeatureConfig, feature_names


@dataclass
class TrainResult:
    bundle: dict
    report: dict


def _grouped_70_20_10(
    groups: np.ndarray, y: np.ndarray | None = None, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign whole animals to a near-70/20/10 stratified split.

    When labels are supplied, deterministic random search balances window totals
    and class proportions while requiring every sensor source in every split.
    """
    rng = np.random.default_rng(random_state)
    train_groups: list[str] = []
    test_groups: list[str] = []
    validation_groups: list[str] = []
    unique = np.unique(groups).astype(str)
    sources = sorted({group.split(":", 1)[0] for group in unique})
    if y is not None:
        n_groups = len(unique)
        n_test = round(n_groups * 0.20)
        n_validation = round(n_groups * 0.10)
        n_train = n_groups - n_test - n_validation
        classes = np.unique(y)
        global_mix = np.array([np.mean(y == label) for label in classes])
        group_sizes = np.array([np.sum(groups == group) for group in unique], dtype=np.int64)
        group_class_counts = np.array([
            [np.sum((groups == group) & (y == label)) for label in classes] for group in unique
        ], dtype=np.int64)
        group_sources = np.array([group.split(":", 1)[0] for group in unique])
        best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
        for _ in range(20_000):
            shuffled = rng.permutation(n_groups)
            candidate_train = shuffled[:n_train]
            candidate_test = shuffled[n_train : n_train + n_test]
            candidate_validation = shuffled[n_train + n_test :]
            parts = (candidate_train, candidate_test, candidate_validation)
            if any(
                not all(np.any(group_sources[part] == source) for source in sources)
                for part in parts
            ):
                continue
            window_counts = np.array([np.sum(group_sizes[part]) for part in parts])
            ratios = window_counts / len(groups)
            score = float(np.sum((ratios - np.array([0.70, 0.20, 0.10])) ** 2) * 20.0)
            for part in parts:
                mix = np.sum(group_class_counts[part], axis=0) / np.sum(group_sizes[part])
                score += float(np.mean((mix - global_mix) ** 2))
            if best is None or score < best[0]:
                best = (score, candidate_train.copy(), candidate_test.copy(), candidate_validation.copy())
        if best is None:
            raise ValueError("could not construct a source-covered grouped 70/20/10 split")
        train_groups, test_groups, validation_groups = unique[best[1]], unique[best[2]], unique[best[3]]
        return (
            np.flatnonzero(np.isin(groups, train_groups)),
            np.flatnonzero(np.isin(groups, test_groups)),
            np.flatnonzero(np.isin(groups, validation_groups)),
        )

    for source in sources:
        source_groups = unique[np.char.startswith(unique, f"{source}:")].copy()
        rng.shuffle(source_groups)
        count = len(source_groups)
        n_test = max(1, round(count * 0.20))
        n_validation = max(1, round(count * 0.10))
        if count - n_test - n_validation < 1:
            raise ValueError(f"source {source!r} needs at least three animals for grouped splitting")
        test_groups.extend(source_groups[:n_test])
        validation_groups.extend(source_groups[n_test : n_test + n_validation])
        train_groups.extend(source_groups[n_test + n_validation :])
    train_idx = np.flatnonzero(np.isin(groups, train_groups))
    test_idx = np.flatnonzero(np.isin(groups, test_groups))
    validation_idx = np.flatnonzero(np.isin(groups, validation_groups))
    return train_idx, test_idx, validation_idx


def _summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _labels_from_probabilities(classes: np.ndarray, probabilities: np.ndarray, threshold: float) -> np.ndarray:
    classes = np.asarray(classes)
    if "Walking" not in classes:
        return classes[np.argmax(probabilities, axis=1)]
    walking_index = int(np.flatnonzero(classes == "Walking")[0])
    non_walking_indices = np.flatnonzero(classes != "Walking")
    predictions = classes[non_walking_indices[np.argmax(probabilities[:, non_walking_indices], axis=1)]]
    predictions[probabilities[:, walking_index] >= threshold] = "Walking"
    return predictions


def _predict_with_walking_threshold(classifier, X: np.ndarray, threshold: float) -> np.ndarray:
    return _labels_from_probabilities(
        classifier.classes_, classifier.predict_proba(X), threshold
    )


def train_bundle(X: np.ndarray, y: np.ndarray, groups: np.ndarray, config: FeatureConfig) -> TrainResult:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("animal-wise validation requires at least two different animals")
    train_idx, test_idx, validation_idx = _grouped_70_20_10(groups, y)
    candidate_templates = {
        "strict_generalization_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=60, max_depth=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "regularized_random_forest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=4, max_depth=24, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
    }
    validation_results = {}
    selected_models = {}
    selected_walking_thresholds = {}
    profile_classifiers = {}
    predictions = np.empty(len(test_idx), dtype=object)
    test_groups_array = groups[test_idx].astype(str)
    sources = sorted({group.split(":", 1)[0] for group in np.unique(groups).astype(str)})
    for source in sources:
        train_source = train_idx[np.char.startswith(groups[train_idx].astype(str), f"{source}:")]
        validation_source = validation_idx[np.char.startswith(groups[validation_idx].astype(str), f"{source}:")]
        test_positions = np.flatnonzero(np.char.startswith(test_groups_array, f"{source}:"))
        validation_results[source] = {}
        trained_candidates = {}
        candidate_thresholds = {}
        for name, template in candidate_templates.items():
            candidate = clone(template)
            candidate.fit(X[train_source], y[train_source])
            validation_probabilities = candidate.predict_proba(X[validation_source])
            best_threshold = 0.5
            best_metrics = None
            for threshold in np.linspace(0.05, 0.50, 46):
                validation_prediction = _labels_from_probabilities(
                    candidate.classes_, validation_probabilities, float(threshold)
                )
                metrics = _summary(y[validation_source], validation_prediction)
                if best_metrics is None or (metrics["macro_f1"], metrics["balanced_accuracy"]) > (
                    best_metrics["macro_f1"], best_metrics["balanced_accuracy"]
                ):
                    best_metrics = metrics
                    best_threshold = float(threshold)
            validation_results[source][name] = {**best_metrics, "walking_threshold": best_threshold}
            training_prediction = _predict_with_walking_threshold(
                candidate, X[train_source], best_threshold
            )
            training_accuracy = float(accuracy_score(y[train_source], training_prediction))
            validation_results[source][name]["train_accuracy"] = training_accuracy
            validation_results[source][name]["generalization_gap"] = max(
                0.0, training_accuracy - best_metrics["accuracy"]
            )
            candidate_thresholds[name] = best_threshold
            trained_candidates[name] = candidate
        eligible = [
            name for name, metrics in validation_results[source].items()
            if metrics["generalization_gap"] <= 0.03
        ]
        selection_pool = eligible or [
            min(
                validation_results[source],
                key=lambda name: validation_results[source][name]["generalization_gap"],
            )
        ]
        selected_name = max(
            selection_pool,
            key=lambda name: (
                validation_results[source][name]["macro_f1"],
                validation_results[source][name]["accuracy"],
            ),
        )
        selected_models[source] = selected_name
        selected_walking_thresholds[source] = candidate_thresholds[selected_name]
        classifier = trained_candidates[selected_name]
        profile_classifiers[source] = classifier
        predictions[test_positions] = _predict_with_walking_threshold(
            classifier, X[test_idx[test_positions]], candidate_thresholds[selected_name]
        )
    profile_test_results = {}
    for source in sources:
        positions = np.flatnonzero(np.char.startswith(test_groups_array, f"{source}:"))
        source_truth = y[test_idx[positions]]
        source_predictions = predictions[positions]
        profile_test_results[source] = {
            "windows": int(len(positions)),
            **_summary(source_truth, source_predictions),
            "classification_report": classification_report(
                source_truth, source_predictions, output_dict=True, zero_division=0
            ),
        }
    class_order = sorted(np.unique(y).tolist())

    gait_detectors = {}
    gait_thresholds = {}
    for source in sources:
        source_train = train_idx[np.char.startswith(groups[train_idx].astype(str), f"{source}:")]
        walking_train = X[source_train][y[source_train] == "Walking"]
        if len(walking_train) < 50:
            continue
        detector = IsolationForest(n_estimators=250, contamination="auto", random_state=42, n_jobs=-1)
        detector.fit(walking_train)
        gait_detectors[source] = detector
        gait_thresholds[source] = float(np.quantile(detector.decision_function(walking_train), 0.03))

    pooled_summary = _summary(y[test_idx], predictions)
    primary_source = "cabritrack"
    primary_positions = np.flatnonzero(np.char.startswith(test_groups_array, f"{primary_source}:"))
    primary_truth = y[test_idx[primary_positions]]
    primary_predictions = predictions[primary_positions]
    primary_summary = _summary(primary_truth, primary_predictions)
    report = {
        "split": "animal/source-grouped 70% training, 20% final testing, 10% validation",
        "split_policy": "Model selected on validation animals with a maximum 3 percentage-point train/validation accuracy gap; final test animals were not used for selection or fitting.",
        "overfitting_policy": {
            "metric": "training accuracy minus unseen-animal validation accuracy",
            "maximum_allowed_gap": 0.03,
            "fallback": "If no candidate passes, select the smallest-gap candidate and report the failure.",
        },
        "requested_ratios": {"training": 0.70, "testing": 0.20, "validation": 0.10},
        "actual_animal_ratios": {
            "training": len(np.unique(groups[train_idx])) / len(unique_groups),
            "testing": len(np.unique(groups[test_idx])) / len(unique_groups),
            "validation": len(np.unique(groups[validation_idx])) / len(unique_groups),
        },
        "train_animals": sorted(np.unique(groups[train_idx]).tolist()),
        "test_animals": sorted(np.unique(groups[test_idx]).tolist()),
        "validation_animals": sorted(np.unique(groups[validation_idx]).tolist()),
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "validation_windows": int(len(validation_idx)),
        "actual_window_ratios": {
            "training": len(train_idx) / len(y),
            "testing": len(test_idx) / len(y),
            "validation": len(validation_idx) / len(y),
        },
        "selected_models_by_sensor_profile": selected_models,
        "walking_thresholds_by_sensor_profile": selected_walking_thresholds,
        "validation_model_selection": validation_results,
        "primary_deployment_profile": "cabritrack (horn, all four target behaviours)",
        "test_by_sensor_profile": profile_test_results,
        "pooled_test_metrics": pooled_summary,
        "primary_test_windows": int(len(primary_positions)),
        **primary_summary,
        "classification_report": classification_report(
            primary_truth, primary_predictions, output_dict=True, zero_division=0
        ),
        "labels": class_order,
        "confusion_matrix": confusion_matrix(primary_truth, primary_predictions, labels=class_order).tolist(),
        "mobility_validation": "Unsupervised normal-walking baseline only; no labeled lame goats were available.",
        "normal_walking_threshold_quantile": 0.03,
    }
    bundle = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_config": config.to_dict(),
        "feature_names": feature_names(config.channels),
        "behavior_classifier": profile_classifiers["cabritrack"],
        "behavior_classifiers": profile_classifiers,
        "gait_detector": gait_detectors["cabritrack"],
        "gait_threshold": gait_thresholds["cabritrack"],
        "gait_detectors": gait_detectors,
        "gait_thresholds": gait_thresholds,
        "classes": class_order,
        "selected_models_by_sensor_profile": selected_models,
        "walking_thresholds": selected_walking_thresholds,
        "data_split": "70% animal groups training / 20% testing / 10% validation",
        "notes": {
            "walking": "CabriTrack Displacement (walking/running/transitions) is mapped to Walking.",
            "mobility_anomaly": "An anomaly flag, not a foot-rot diagnosis; requires local labeled validation.",
            "piezo": "Supported by refinement/inference schema but not present in the public pretraining data.",
        },
    }
    return TrainResult(bundle=bundle, report=report)


def save_bundle(bundle: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
