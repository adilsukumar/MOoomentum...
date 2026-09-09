import unittest

from vitalsense.longitudinal import BaselineMetric, PersonalBaseline
from vitalsense.models import AnalysisConfig
from vitalsense.pipeline import analyze_session
from vitalsense.synthetic import generate_synthetic_session


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_session(
            generate_synthetic_session(
                duration_s=360.0,
                sample_rate_hz=100.0,
                heart_rate_bpm=72.0,
                respiratory_rate_bpm=15.0,
                include_activity=False,
            )
        )

    def test_detects_rates(self):
        heart = self.result["summary"]["heart_rate_bpm"]["median"]
        respiration = self.result["summary"]["respiratory_rate_bpm"]["median"]
        self.assertAlmostEqual(heart, 72.0, delta=1.0)
        self.assertAlmostEqual(respiration, 15.0, delta=0.5)

    def test_emits_all_metric_families(self):
        summary = self.result["summary"]
        self.assertIn("aggregate_interbeat_metrics", summary)
        self.assertIn("aggregate_respiratory_variability", summary)
        self.assertIn("aggregate_cardiorespiratory_coupling", summary)
        self.assertGreater(summary["sleep_candidate_minutes"], 0.0)
        self.assertGreater(summary["usable_data_coverage_pct"], 90.0)

    def test_emits_frontend_ready_values_and_bases(self):
        frontend = self.result["frontend"]
        self.assertEqual(frontend["schema_version"], "vitalsense.frontend.v1")
        self.assertIn("heart_rate_bpm", frontend["metrics"])
        self.assertIn("ibi_rmssd_ms", frontend["metrics"])
        self.assertIn("heart_rate_recovery_10m_bpm", frontend["metrics"])
        heart = frontend["metrics"]["heart_rate_bpm"]
        self.assertTrue(heart["available"])
        self.assertEqual(heart["unit"], "bpm")
        self.assertIsNotNone(heart["base"]["lower"])
        self.assertTrue(heart["thresholds"])

    def test_eligible_personal_baseline_replaces_population_base(self):
        baseline = PersonalBaseline(
            dog_id="dog-1",
            context="sleep_candidate",
            baseline_days=14,
            eligible=True,
            metrics={
                "heart_rate_bpm": BaselineMetric(72.0, 2.0, 150, 14),
                "respiratory_rate_bpm": BaselineMetric(15.0, 1.0, 150, 14),
            },
        )
        result = analyze_session(
            generate_synthetic_session(duration_s=180.0, include_activity=False), baseline=baseline
        )
        base = result["frontend"]["metrics"]["heart_rate_bpm"]["base"]
        self.assertEqual(base["kind"], "personal_robust_band")
        self.assertEqual(base["center"], 72.0)
        self.assertEqual(base["lower"], 66.0)
        self.assertEqual(base["upper"], 78.0)

    def test_piezo_press_artifact_is_not_reported_as_hr_spike(self):
        batch = generate_synthetic_session(duration_s=300.0, include_activity=False)
        batch.piezo[15000] += 100.0
        result = analyze_session(batch)
        affected = [
            window for window in result["windows"]
            if "piezo_impulse_artifact_candidate" in window["reason_codes"]
        ]
        self.assertTrue(affected)
        self.assertTrue(all(not window["valid"] for window in affected))
        self.assertGreater(result["summary"]["piezo_impulse_artifact_window_pct"], 0.0)
        self.assertNotIn("HR_HIGH_180", {alert["rule_id"] for alert in result["alerts"]})

    def test_normal_session_has_no_alert(self):
        self.assertEqual(self.result["alerts"], [])

    def test_high_resting_rate_alert(self):
        result = analyze_session(
            generate_synthetic_session(
                duration_s=240.0,
                sample_rate_hz=100.0,
                heart_rate_bpm=150.0,
                respiratory_rate_bpm=15.0,
                include_activity=False,
            )
        )
        rules = {alert["rule_id"] for alert in result["alerts"]}
        self.assertIn("HR_HIGH_140", rules)

    def test_very_low_resting_rate_alert(self):
        result = analyze_session(
            generate_synthetic_session(
                duration_s=180.0,
                sample_rate_hz=100.0,
                heart_rate_bpm=25.0,
                respiratory_rate_bpm=12.0,
                include_activity=False,
            )
        )
        rules = {alert["rule_id"] for alert in result["alerts"]}
        self.assertIn("HR_LOW_30", rules)

    def test_elevated_sleeping_respiration_watch(self):
        result = analyze_session(
            generate_synthetic_session(
                duration_s=360.0,
                sample_rate_hz=100.0,
                heart_rate_bpm=72.0,
                respiratory_rate_bpm=28.0,
                include_activity=False,
            )
        )
        rules = {alert["rule_id"] for alert in result["alerts"]}
        self.assertIn("RR_SLEEP_25", rules)
        self.assertNotIn("RR_SLEEP_30", rules)

    def test_custom_quality_config_is_accepted(self):
        result = analyze_session(
            generate_synthetic_session(duration_s=120.0, include_activity=False),
            config=AnalysisConfig(minimum_fusion_quality=0.2),
        )
        self.assertGreater(result["summary"]["valid_window_count"], 0)


if __name__ == "__main__":
    unittest.main()
