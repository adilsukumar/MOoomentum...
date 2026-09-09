import math
import unittest

from vitalsense.metrics import cardiorespiratory_coupling, interval_variability, respiratory_variability


class MetricsTests(unittest.TestCase):
    def test_interbeat_variability(self):
        result = interval_variability([0.8, 0.9, 0.8, 0.9], prefix="ibi")
        self.assertAlmostEqual(result["mean_ibi_ms"], 850.0)
        self.assertAlmostEqual(result["ibi_rmssd_ms"], 100.0)
        self.assertAlmostEqual(result["ibi_pnn50_pct"], 100.0)
        self.assertAlmostEqual(result["ibi_poincare_sd1_ms"], 100.0 / math.sqrt(2.0))

    def test_respiratory_variability(self):
        result = respiratory_variability([4.0, 4.2, 3.8, 4.0])
        self.assertAlmostEqual(result["mean_breath_interval_s"], 4.0)
        self.assertGreater(result["breath_interval_cv"], 0.0)
        self.assertGreaterEqual(result["irregular_breathing_index"], 0.0)
        self.assertLessEqual(result["irregular_breathing_index"], 1.0)

    def test_coupling_ratio(self):
        result = cardiorespiratory_coupling(
            pulse_times_s=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            breath_times_s=[0.0, 4.0, 8.0],
            heart_rate_bpm=60.0,
            respiratory_rate_bpm=15.0,
        )
        self.assertAlmostEqual(result["beats_per_breath"], 4.0)
        self.assertIsNotNone(result["respiratory_phase_locking_value"])


if __name__ == "__main__":
    unittest.main()

