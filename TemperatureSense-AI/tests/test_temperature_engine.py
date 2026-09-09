import unittest

from smart_collar_temperature import (
    ActivityLevel,
    BodyCondition,
    CoatType,
    DividerOrientation,
    DogProfile,
    EnvironmentContext,
    RabiesContext,
    SamplingPolicy,
    Severity,
    SizeClass,
    SleepState,
    TemperatureScheduler,
    ThermistorConfig,
    adc_to_celsius,
    assess_environment,
    assess_rabies_context,
    assess_surface_contact_temperature,
    get_breed_evidence,
)


class ThermistorTests(unittest.TestCase):
    def test_half_scale_is_about_25_c_with_matching_divider(self):
        result = adc_to_celsius(2048)
        self.assertAlmostEqual(result, 25.0, delta=0.1)

    def test_other_divider_orientation(self):
        config = ThermistorConfig(
            orientation=DividerOrientation.NTC_TO_VREF_FIXED_TO_GROUND
        )
        result = adc_to_celsius(2048, config)
        self.assertAlmostEqual(result, 25.0, delta=0.1)

    def test_saturated_adc_is_rejected(self):
        with self.assertRaises(ValueError):
            adc_to_celsius(0)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = TemperatureScheduler(SamplingPolicy())

    def test_awake_reads_environment_only(self):
        result = self.scheduler.schedule(
            now_s=100,
            bmi270_state=SleepState.AWAKE,
            state_since_s=0,
            last_sht40_s=0,
            last_ntc_s=None,
        )
        self.assertTrue(result.read_sht40)
        self.assertFalse(result.read_ntc)

    def test_unconfirmed_sleep_does_not_read_surface_ntc(self):
        result = self.scheduler.schedule(
            now_s=100,
            bmi270_state=SleepState.SLEEPING,
            state_since_s=0,
            last_sht40_s=0,
            last_ntc_s=None,
        )
        self.assertFalse(result.sleep_confirmed)
        self.assertFalse(result.read_ntc)

    def test_confirmed_sleep_reads_ntc_and_powers_down_sht40(self):
        result = self.scheduler.schedule(
            now_s=600,
            bmi270_state=SleepState.SLEEPING,
            state_since_s=0,
            last_sht40_s=550,
            last_ntc_s=500,
        )
        self.assertTrue(result.sleep_confirmed)
        self.assertTrue(result.read_ntc)
        self.assertFalse(result.read_sht40)

    def test_optional_sleep_environment_heartbeat(self):
        scheduler = TemperatureScheduler(
            SamplingPolicy(sleeping_environment_interval_s=300)
        )
        result = scheduler.schedule(
            now_s=600,
            bmi270_state=SleepState.SLEEPING,
            state_since_s=0,
            last_sht40_s=200,
            last_ntc_s=500,
        )
        self.assertTrue(result.read_sht40)


class SafetyTests(unittest.TestCase):
    def test_moderate_environment_is_normal(self):
        result = assess_environment(22, 40, DogProfile())
        self.assertEqual(result.severity, Severity.NORMAL)

    def test_heat_humidity_and_activity_escalate(self):
        dog = DogProfile(
            breed="Husky",
            size=SizeClass.LARGE,
            coat=CoatType.HEAVY_DOUBLE,
            body_condition=BodyCondition.IDEAL,
        )
        result = assess_environment(
            30,
            85,
            dog,
            EnvironmentContext(activity=ActivityLevel.ACTIVE, direct_sun=True),
        )
        self.assertEqual(result.severity, Severity.CRITICAL)

    def test_invalid_environment_input(self):
        result = assess_environment(22, 150, DogProfile())
        self.assertEqual(result.severity, Severity.INVALID)

    def test_cold_alert_never_recommends_cooling(self):
        dog = DogProfile(coat=CoatType.HAIRLESS, size=SizeClass.TOY)
        result = assess_environment(5, 50, dog)
        self.assertEqual(result.severity, Severity.CRITICAL)
        self.assertTrue(any("warm" in action for action in result.actions))
        self.assertFalse(any("cooling" in action for action in result.actions))

    def test_surface_temperature_is_blocked_while_awake(self):
        dog = DogProfile(healthy_sleep_surface_baseline_c=33.0)
        result = assess_surface_contact_temperature(
            [33.0] * 5, dog, sleep_confirmed=False
        )
        self.assertEqual(result.code, "SURFACE_NOT_SLEEPING")

    def test_surface_temperature_needs_personal_baseline(self):
        result = assess_surface_contact_temperature(
            [33.0] * 5, DogProfile(), sleep_confirmed=True
        )
        self.assertEqual(result.code, "SURFACE_BASELINE_REQUIRED")

    def test_stable_elevated_surface_trend(self):
        dog = DogProfile(healthy_sleep_surface_baseline_c=33.0)
        result = assess_surface_contact_temperature(
            [34.0, 34.1, 34.0, 34.1, 34.0], dog, sleep_confirmed=True
        )
        self.assertEqual(result.code, "SURFACE_TREND_ELEVATED")

    def test_unstable_contact_is_rejected(self):
        dog = DogProfile(healthy_sleep_surface_baseline_c=33.0)
        result = assess_surface_contact_temperature(
            [32.0, 32.5, 33.0, 33.5, 34.0], dog, sleep_confirmed=True
        )
        self.assertEqual(result.code, "SURFACE_UNSTABLE_CONTACT")

    def test_surface_anomaly_alone_never_predicts_rabies(self):
        dog = DogProfile(healthy_sleep_surface_baseline_c=33.0)
        surface = assess_surface_contact_temperature(
            [34.0] * 5, dog, sleep_confirmed=True
        )
        result = assess_rabies_context(RabiesContext(), surface)
        self.assertEqual(result.code, "NO_RABIES_INFERENCE")

    def test_possible_exposure_is_critical_without_waiting_for_temperature(self):
        result = assess_rabies_context(
            RabiesContext(known_or_possible_exposure=True)
        )
        self.assertEqual(result.severity, Severity.CRITICAL)

    def test_breed_alias_finds_population_evidence(self):
        evidence = get_breed_evidence("Frenchie")
        self.assertIsNotNone(evidence)
        self.assertTrue(evidence.significantly_higher)


if __name__ == "__main__":
    unittest.main()
