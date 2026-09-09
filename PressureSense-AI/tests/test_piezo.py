import numpy as np

from src.goat_motion.piezo import analyze_piezo_window


def test_piezo_states():
    assert analyze_piezo_window(np.zeros(50), baseline=0)["state"] == "normal"
    assert analyze_piezo_window(np.array([0, 180, 0]), baseline=0)["state"] == "pressed"
    hard = analyze_piezo_window(np.array([0, 900, 0]), baseline=0)
    assert hard["state"] == "abnormal_force"
    assert hard["alert"] is True


def test_repeated_presses_are_counted():
    result = analyze_piezo_window(np.array([0, 200, 0, 220, 0]), baseline=0)
    assert result["spike_count"] == 2
