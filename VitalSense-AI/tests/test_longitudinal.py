import unittest

from vitalsense.longitudinal import build_personal_baseline, evaluate_multiday_trends, score_against_baseline


class LongitudinalTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for day in range(10):
            for sample in range(12):
                self.records.append(
                    {
                        "date": f"2026-08-{day + 1:02d}",
                        "context": "sleep_candidate",
                        "heart_rate_bpm": 60.0 + ((sample % 3) - 1),
                        "respiratory_rate_bpm": 15.0 + ((sample % 3) - 1) * 0.25,
                    }
                )

    def test_builds_eligible_baseline(self):
        baseline = build_personal_baseline(self.records, dog_id="dog-1")
        self.assertTrue(baseline.eligible)
        self.assertAlmostEqual(baseline.metrics["heart_rate_bpm"].median, 60.0)
        self.assertAlmostEqual(baseline.metrics["respiratory_rate_bpm"].median, 15.0)

    def test_scores_change(self):
        baseline = build_personal_baseline(self.records, dog_id="dog-1")
        summary = {
            "heart_rate_bpm": {"median": 78.0},
            "respiratory_rate_bpm": {"median": 21.0},
        }
        score = score_against_baseline(summary, baseline)
        self.assertTrue(score["eligible"])
        self.assertGreater(score["metrics"]["heart_rate_bpm"]["relative_change_pct"], 20.0)
        self.assertGreater(score["overall_anomaly_score"], 0.0)

    def test_persistent_multiday_trend(self):
        baseline = build_personal_baseline(self.records, dog_id="dog-1")
        shifted = []
        for day in range(8):
            shifted.append(
                {
                    "date": f"2026-09-{day + 1:02d}",
                    "context": "sleep_candidate",
                    "heart_rate_bpm": 78.0,
                    "respiratory_rate_bpm": 20.0,
                }
            )
        trend = evaluate_multiday_trends(shifted, baseline)
        rules = {alert["rule_id"] for alert in trend["alerts"]}
        self.assertIn("PERSISTENT_3DAY_BASELINE_SHIFT", rules)


if __name__ == "__main__":
    unittest.main()
