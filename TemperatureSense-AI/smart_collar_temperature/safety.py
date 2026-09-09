from __future__ import annotations

import math
import statistics

from .breeds import get_breed_evidence
from .models import (
    ActivityLevel,
    Alert,
    BodyCondition,
    CoatType,
    DogProfile,
    EnvironmentContext,
    RabiesContext,
    Severity,
    SizeClass,
    SkullShape,
)


def _heat_vulnerability(profile: DogProfile) -> tuple[int, list[str]]:
    points = 0
    reasons: list[str] = []
    if profile.skull is SkullShape.BRACHYCEPHALIC:
        points += 3
        reasons.append("brachycephalic airway anatomy")
    if profile.coat in (CoatType.DOUBLE, CoatType.HEAVY_DOUBLE):
        points += 1
        reasons.append("double/heavy coat")
    if profile.size is SizeClass.GIANT:
        points += 1
        reasons.append("giant body size")
    if profile.body_condition is BodyCondition.OVERWEIGHT:
        points += 1
        reasons.append("overweight")
    elif profile.body_condition is BodyCondition.OBESE:
        points += 2
        reasons.append("obesity")
    if profile.age_years is not None and (
        profile.age_years < 1.0 or profile.age_years >= 8.0
    ):
        points += 1
        reasons.append("young or older age")
    if profile.respiratory_or_cardiac_disease:
        points += 3
        reasons.append("respiratory/cardiac condition")

    evidence = get_breed_evidence(profile.breed)
    if evidence and evidence.significantly_higher:
        points += 2 if evidence.odds_ratio_vs_labrador >= 5.0 else 1
        reasons.append("breed had higher population odds of heat-related illness")
    if profile.heat_acclimated:
        points = max(0, points - 1)
    return points, reasons


def _cold_vulnerability(profile: DogProfile) -> tuple[int, list[str]]:
    points = 0
    reasons: list[str] = []
    if profile.coat is CoatType.HAIRLESS:
        points += 3
        reasons.append("hairless coat")
    elif profile.coat in (CoatType.SHORT_SINGLE, CoatType.LONG_SINGLE):
        points += 1
        reasons.append("single coat")
    elif profile.coat is CoatType.DOUBLE:
        points -= 1
    elif profile.coat is CoatType.HEAVY_DOUBLE:
        points -= 2
    if profile.size is SizeClass.TOY:
        points += 2
        reasons.append("toy size")
    elif profile.size is SizeClass.SMALL:
        points += 1
        reasons.append("small size")
    if profile.body_condition is BodyCondition.UNDERWEIGHT:
        points += 1
        reasons.append("underweight")
    if profile.age_years is not None and (
        profile.age_years < 1.0 or profile.age_years >= 8.0
    ):
        points += 1
        reasons.append("young or older age")
    if profile.cold_acclimated:
        points -= 1
    return points, reasons


def _environment_base_level(temperature_c: float) -> tuple[int, str, str]:
    # These are conservative product alert bands, not veterinary 'safe limits'.
    if temperature_c >= 35.0:
        return 3, "extreme heat range", "ENV_EXTREME_HEAT"
    if temperature_c >= 30.0:
        return 2, "hot range", "ENV_HOT"
    if temperature_c >= 25.0:
        return 1, "warm range", "ENV_WARM"
    if temperature_c < -5.0:
        return 3, "extreme cold range", "ENV_EXTREME_COLD"
    if temperature_c < 0.0:
        return 2, "very cold range", "ENV_VERY_COLD"
    if temperature_c < 10.0:
        return 1, "cold range", "ENV_COLD"
    return 0, "moderate range", "ENV_MODERATE"


def assess_environment(
    temperature_c: float,
    relative_humidity_pct: float,
    profile: DogProfile,
    context: EnvironmentContext = EnvironmentContext(),
) -> Alert:
    """Classify local environmental exposure measured by the SHT40.

    Ambient temperature alone cannot prove that conditions are safe.  The
    SHT40 also cannot measure sun/radiant heat, ground temperature, wind, or a
    hot vehicle's spatial extremes, so context deliberately increases risk.
    """

    if (
        not math.isfinite(temperature_c)
        or not math.isfinite(relative_humidity_pct)
        or not -40.0 <= temperature_c <= 125.0
        or not 0.0 <= relative_humidity_pct <= 100.0
    ):
        return Alert(
            Severity.INVALID,
            "ENV_SENSOR_INVALID",
            "unavailable",
            "Environmental reading failed plausibility checks.",
            ("Check SHT40 placement, CRC handling, and sensor connection.",),
        )

    level, range_label, code = _environment_base_level(temperature_c)
    reasons: list[str] = []
    heat_points, heat_reasons = _heat_vulnerability(profile)
    cold_points, cold_reasons = _cold_vulnerability(profile)

    if temperature_c >= 20.0:
        if relative_humidity_pct >= 80.0 and temperature_c >= 24.0:
            level += 1
            reasons.append("very high humidity limits evaporative cooling")
        elif relative_humidity_pct >= 65.0 and temperature_c >= 24.0:
            level += 1
            reasons.append("high humidity reduces panting efficiency")
        if context.activity is ActivityLevel.ACTIVE:
            level += 1
            reasons.append("active movement adds metabolic heat")
        elif context.activity is ActivityLevel.INTENSE:
            level += 2
            reasons.append("intense activity adds substantial metabolic heat")
        if context.direct_sun:
            level += 1
            reasons.append("direct sun/radiant heat is not measured by SHT40")
        if context.poor_airflow:
            level += 1
            reasons.append("poor airflow impairs cooling")
        if heat_points >= 7:
            level += 2
            reasons.extend(heat_reasons)
        elif heat_points >= 3:
            level += 1
            reasons.extend(heat_reasons)
    elif temperature_c < 10.0:
        if cold_points >= 4:
            level += 2
            reasons.extend(cold_reasons)
        elif cold_points >= 1:
            level += 1
            reasons.extend(cold_reasons)

    if context.enclosed_vehicle and temperature_c >= 20.0:
        level = 3
        code = "ENV_VEHICLE_DANGER"
        range_label = "dangerous enclosed-space range"
        reasons.append("enclosed vehicles can heat rapidly and prevent escape")

    level = min(3, level)
    severity = (Severity.NORMAL, Severity.CAUTION, Severity.HIGH, Severity.CRITICAL)[
        level
    ]
    heat_actions_by_level = {
        Severity.NORMAL: ("Continue monitoring; this is not a guarantee of safety.",),
        Severity.CAUTION: (
            "Reduce exertion and provide water, shade, and airflow.",
            "Watch for heavy panting, shade-seeking, weakness, vomiting, or confusion.",
        ),
        Severity.HIGH: (
            "Stop exertion and move the dog to a cooler, ventilated place.",
            "Begin active cooling with cool water and contact a veterinarian if signs are present.",
        ),
        Severity.CRITICAL: (
            "Remove the dog from the exposure immediately and begin active cooling with cool water.",
            "Seek emergency veterinary care for any heat-illness signs; do not wait for collar temperature.",
        ),
    }
    cold_actions_by_level = {
        Severity.NORMAL: ("Continue monitoring; this is not a guarantee of safety.",),
        Severity.CAUTION: (
            "Reduce exposure and provide a dry, sheltered, warmer place.",
            "Watch for shivering, lifting paws, weakness, confusion, or unusual sleepiness.",
        ),
        Severity.HIGH: (
            "End the cold exposure and warm the dog gradually with dry insulation.",
            "Contact a veterinarian promptly if signs persist or the dog is wet, weak, or confused.",
        ),
        Severity.CRITICAL: (
            "Move to shelter, remove wet materials, wrap in dry blankets, and warm gradually.",
            "Seek emergency veterinary care; avoid direct high heat that can burn the skin.",
        ),
    }
    actions_by_level = cold_actions_by_level if temperature_c < 10.0 else heat_actions_by_level
    return Alert(
        severity,
        code,
        range_label,
        f"Environmental exposure is in the {range_label} for this profile and context.",
        actions_by_level[severity],
        tuple(dict.fromkeys(reasons)),
        {
            "engineering_band_c": (
                "35+" if temperature_c >= 35 else
                "30-34.9" if temperature_c >= 30 else
                "25-29.9" if temperature_c >= 25 else
                "10-24.9" if temperature_c >= 10 else
                "0-9.9" if temperature_c >= 0 else
                "-5--0.1" if temperature_c >= -5 else "below -5"
            ),
            "heat_vulnerability_score": heat_points,
            "cold_vulnerability_score": cold_points,
        },
    )


def assess_surface_contact_temperature(
    surface_samples_c: list[float] | tuple[float, ...],
    profile: DogProfile,
    *,
    sleep_confirmed: bool,
) -> Alert:
    """Assess a stable sleep-only collar surface-contact NTC window.

    The output is strictly a surface-temperature trend relative to this dog's
    healthy baseline at the same sensor site and collar fit. It is never
    converted to internal temperature and cannot establish fever.
    """

    if not sleep_confirmed:
        return Alert(
            Severity.INVALID,
            "SURFACE_NOT_SLEEPING",
            "unavailable",
            "Surface-contact temperature is disabled until MotionSense sleep is stable.",
        )
    valid = [
        float(value)
        for value in surface_samples_c
        if math.isfinite(value) and 0.0 <= value <= 50.0
    ]
    if len(valid) < 5:
        return Alert(
            Severity.INVALID,
            "SURFACE_TOO_FEW_SAMPLES",
            "unavailable",
            "At least five valid contact readings are required.",
            ("Keep the NTC in steady surface contact and collect another window.",),
        )
    contact_span = max(valid) - min(valid)
    if contact_span > 1.0:
        return Alert(
            Severity.INVALID,
            "SURFACE_UNSTABLE_CONTACT",
            "unavailable",
            "The sample window changed too much for a reliable contact reading.",
            ("Check collar fit, fur intrusion, motion, and sensor pressure.",),
            metadata={
                "contact_quality": "unstable",
                "sample_span_c": round(contact_span, 2),
            },
        )
    observed = statistics.median(valid)
    baseline = profile.healthy_sleep_surface_baseline_c
    if baseline is None:
        return Alert(
            Severity.INVALID,
            "SURFACE_BASELINE_REQUIRED",
            "calibration required",
            "A healthy per-dog surface baseline is required before trend alerts are enabled.",
            ("Collect at least seven healthy nights using the same collar fit and site.",),
        )

    delta = observed - baseline
    common_metadata: dict[str, object] = {
        "surface_delta_band_c": (
            "above +1.5" if delta > 1.5 else
            "+0.8 to +1.5" if delta > 0.8 else
            "-0.8 to +0.8" if delta >= -0.8 else
            "-1.5 to -0.8" if delta >= -1.5 else "below -1.5"
        ),
        "trend_direction": (
            "high" if delta > 0.8 else "low" if delta < -0.8 else "baseline"
        ),
        "contact_quality": "stable",
        "sample_count": len(valid),
        "sample_span_c": round(contact_span, 2),
    }

    if delta > 1.5:
        return Alert(
            Severity.HIGH,
            "SURFACE_TREND_HIGH",
            "well above personal sleep baseline",
            "Surface-contact temperature is well above this dog's healthy sleep baseline.",
            (
                "Wake and check the dog, collar contact, and environment.",
                "Contact a veterinarian promptly if the change persists or any illness signs are present.",
            ),
            metadata=common_metadata,
        )
    if delta > 0.8:
        return Alert(
            Severity.CAUTION,
            "SURFACE_TREND_ELEVATED",
            "above personal sleep baseline",
            "Surface-contact temperature is above this dog's healthy sleep baseline.",
            ("Repeat the measurement and check collar contact, symptoms, and environment.",),
            metadata=common_metadata,
        )
    if delta < -1.5:
        return Alert(
            Severity.HIGH,
            "SURFACE_TREND_LOW",
            "well below personal sleep baseline",
            "Surface-contact temperature is well below this dog's healthy sleep baseline.",
            ("Check sensor contact and the dog; seek veterinary advice if confirmed or symptomatic.",),
            metadata=common_metadata,
        )
    if delta < -0.8:
        return Alert(
            Severity.CAUTION,
            "SURFACE_TREND_LOW",
            "below personal sleep baseline",
            "Surface-contact temperature is below this dog's healthy sleep baseline.",
            ("Repeat the measurement and check sensor contact.",),
            metadata=common_metadata,
        )
    return Alert(
        Severity.NORMAL,
        "SURFACE_TREND_BASELINE",
        "personal sleep-baseline range",
        "Surface-contact temperature is within this dog's learned healthy sleep range.",
        ("Continue trend monitoring; this does not establish internal temperature or health.",),
        metadata=common_metadata,
    )


def assess_rabies_context(
    context: RabiesContext, surface_alert: Alert | None = None
) -> Alert:
    """Route rabies-related context without making a sensor diagnosis."""

    surface_anomaly = surface_alert is not None and surface_alert.code in {
        "SURFACE_TREND_HIGH",
        "SURFACE_TREND_ELEVATED",
        "SURFACE_TREND_LOW",
    }
    high_specificity_cluster = context.neurologic_signs and (
        context.swallowing_difficulty_or_excess_salivation
        or context.abrupt_behavior_change
        or context.aggression_or_unusual_tameness
    )
    if context.known_or_possible_exposure or high_specificity_cluster:
        return Alert(
            Severity.CRITICAL,
            "RABIES_CONTEXT_EMERGENCY",
            "rabies exposure/signs require professional assessment",
            "The context warrants immediate veterinary and public-health assessment; the collar cannot diagnose rabies.",
            (
                "Avoid saliva and bites, separate people and animals safely, and do not handle the mouth.",
                "Call a veterinarian and local public-health/animal-control authority immediately.",
                "Anyone bitten or scratched should wash the wound and obtain urgent human medical advice.",
            ),
            reasons=(
                "known/possible exposure" if context.known_or_possible_exposure else "progressive neurologic/sign cluster",
            ),
        )
    if surface_anomaly and (
        context.abrupt_behavior_change
        or context.swallowing_difficulty_or_excess_salivation
        or context.aggression_or_unusual_tameness
    ):
        return Alert(
            Severity.HIGH,
            "ILLNESS_WITH_BEHAVIOR_CHANGE",
            "urgent nonspecific illness pattern",
            "A surface-temperature anomaly plus behavior/oral signs needs urgent veterinary assessment, but is not a rabies result.",
            ("Limit saliva exposure and contact a veterinarian promptly.",),
        )
    return Alert(
        Severity.NORMAL,
        "NO_RABIES_INFERENCE",
        "no rabies inference",
        "Surface-contact temperature alone is nonspecific and must never produce a rabies-positive prediction.",
        ("Use exposure history, vaccination status, veterinary assessment, and official testing protocols.",),
    )
