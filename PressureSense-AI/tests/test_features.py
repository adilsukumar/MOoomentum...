import numpy as np

from src.goat_motion.features import FeatureConfig, extract_window_features, feature_names
from src.goat_motion.model import _grouped_70_20_10


def test_feature_shape_and_finiteness():
    config = FeatureConfig(sample_rate_hz=25, window_seconds=5)
    rng = np.random.default_rng(42)
    values = rng.normal(size=(config.window_samples, 3))
    features = extract_window_features(values, config)
    assert features.shape == (len(feature_names(config.channels)),)
    assert np.isfinite(features).all()


def test_rejects_short_windows():
    config = FeatureConfig()
    try:
        extract_window_features(np.zeros((4, 3)), config)
    except ValueError as exc:
        assert "at least 8" in str(exc)
    else:
        raise AssertionError("short window was accepted")


def test_grouped_split_has_no_animal_leakage():
    groups = np.array([f"source:{animal}" for animal in range(10) for _ in range(3)])
    train, test, validation = _grouped_70_20_10(groups)
    train_groups = set(groups[train])
    test_groups = set(groups[test])
    validation_groups = set(groups[validation])
    assert len(train_groups) == 7
    assert len(test_groups) == 2
    assert len(validation_groups) == 1
    assert train_groups.isdisjoint(test_groups | validation_groups)
    assert test_groups.isdisjoint(validation_groups)
