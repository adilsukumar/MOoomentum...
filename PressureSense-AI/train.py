from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.goat_motion.data import load_cabritrack_features, load_mosar_features, load_zenodo_features
from src.goat_motion.features import FeatureConfig
from src.goat_motion.model import save_bundle, train_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train goat behaviour + normal-gait baseline")
    parser.add_argument("--cabritrack", required=True, help="CabriTrack Final-DataPaper-Horn.txt")
    parser.add_argument("--mosar-dir", help="Directory containing MoSAR raw goat CSV files")
    parser.add_argument("--zenodo-dir", help="Directory containing UPV Zenodo LIS3DH CSV files")
    parser.add_argument("--model", default="models/goat_motion.joblib")
    parser.add_argument("--report", default="reports/metrics.json")
    parser.add_argument("--sample-rate", type=float, default=25.0)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--max-windows-per-class", type=int, default=12000)
    parser.add_argument("--feature-cache", default="data/processed/training_features.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    config = FeatureConfig(sample_rate_hz=args.sample_rate, window_seconds=args.window_seconds)
    cache_path = Path(args.feature_cache)
    if cache_path.exists() and not args.rebuild_cache:
        cached = np.load(cache_path)
        X, y, groups = cached["X"], cached["y"], cached["groups"]
    else:
        X, y, groups = load_cabritrack_features(
            args.cabritrack, config, max_windows_per_class=args.max_windows_per_class
        )
        if args.mosar_dir:
            mosar_X, mosar_y, mosar_groups = load_mosar_features(args.mosar_dir, config)
            X = np.vstack([X, mosar_X])
            y = np.concatenate([y, mosar_y])
            groups = np.concatenate([groups, mosar_groups])
        if args.zenodo_dir:
            zenodo_X, zenodo_y, zenodo_groups = load_zenodo_features(
                args.zenodo_dir, config, max_windows_per_class=args.max_windows_per_class
            )
            X = np.vstack([X, zenodo_X])
            y = np.concatenate([y, zenodo_y])
            groups = np.concatenate([groups, zenodo_groups])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, groups=groups)
    result = train_bundle(X, y, groups, config)
    save_bundle(result.bundle, args.model)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
