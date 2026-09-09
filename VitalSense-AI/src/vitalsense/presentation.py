"""Stable, frontend-ready view model for VitalSense analysis results.

This module deliberately keeps population reference intervals separate from
alert guardrails.  The published values used here are descriptive cohort
statistics (mostly IQRs), not diagnostic limits.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .models import AnalysisConfig, DogProfile, WindowResult


_SEVERITY_RANK = {"none": 0, "info": 1, "watch": 2, "high": 3, "urgent": 4}


def build_frontend_payload(
    summary: dict[str, Any],
    windows: list[WindowResult],
    profile: DogProfile,
    baseline_score: dict[str, Any],
    alerts: list[dict[str, Any]],
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Return a compact contract that a web or mobile frontend can render directly."""
    reference = _reference_for_profile(profile)
    eligible_rest = [
        window
        for window in windows
        if window.valid and window.context in {"sleep_candidate", "quiet_awake_rest"}
    ]
    source_windows = eligible_rest or [window for window in windows if window.valid]
    heart_rate = _median(window.heart_rate_bpm for window in source_windows)
    respiratory_rate = _median(window.respiratory_rate_bpm for window in source_windows)
    if heart_rate is None:
        heart_rate = _nested(summary, "heart_rate_bpm", "median")
    if respiratory_rate is None:
        respiratory_rate = _nested(summary, "respiratory_rate_bpm", "median")

    heart_base = _baseline_for_metric("heart_rate_bpm", baseline_score, reference["heart_rate_bpm"])
    resp_base = _baseline_for_metric("respiratory_rate_bpm", baseline_score, reference["respiratory_rate_bpm"])
    metrics: dict[str, dict[str, Any]] = {}

    def add(
        metric_id: str,
        label: str,
        category: str,
        value: float | int | str | None,
        unit: str,
        decimals: int = 1,
        *,
        base: dict[str, Any] | None = None,
        thresholds: list[dict[str, Any]] | None = None,
        state: str | None = None,
        severity: str | None = None,
        confidence_pct: float | None = None,
        description: str | None = None,
    ) -> None:
        numeric_value = _finite(value)
        available = value is not None and (not isinstance(value, (float, np.floating)) or numeric_value is not None)
        final_value: float | int | str | None = numeric_value if isinstance(value, (float, np.floating)) else value
        metric_base = base or _unestablished_base()
        final_state = state or _reference_state(numeric_value, metric_base)
        final_severity = severity or _metric_alert_severity(metric_id, alerts)
        if not available:
            final_state = "unavailable"
            final_severity = "unavailable"
        metrics[metric_id] = {
            "label": label,
            "category": category,
            "value": final_value,
            "unit": unit,
            "display_value": _display(final_value, unit, decimals),
            "decimals": decimals,
            "available": available,
            "state": final_state,
            "severity": final_severity,
            "confidence_pct": _round_or_none(confidence_pct, 0),
            "base": metric_base,
            "thresholds": thresholds or [],
            "description": description,
        }

    fusion_pct = _percent(_nested(summary, "fusion_quality", "median"))
    hr_thresholds = [
        _threshold("below", 30.0, "urgent", "30 continuous seconds, high quality"),
        _threshold("below", 40.0, "high", "3 of 5 valid 60-second windows"),
        _threshold("above", 140.0, "high", "3 of 5 valid 60-second windows"),
        _threshold("above", 180.0, "urgent", "30 continuous seconds, high quality"),
    ]
    rr_thresholds = [
        _threshold("below", 8.0, "info", "2 valid 60-second windows"),
        _threshold("above", 25.0, "watch", "2 nights / repeated sleeping readings"),
        _threshold("at_or_above", 30.0, "high", "2 valid windows separated by 10 minutes"),
        _threshold("at_or_above", 40.0, "urgent", "5 continuous valid minutes"),
    ]
    add(
        "heart_rate_bpm", "Mechanical heart rate", "vitals", heart_rate, "bpm", 0,
        base=heart_base, thresholds=hr_thresholds, confidence_pct=fusion_pct,
        description="Piezo-derived mechanical pulse rate; not ECG.",
    )
    add(
        "respiratory_rate_bpm", "Respiratory rate", "vitals", respiratory_rate, "breaths/min", 1,
        base=resp_base, thresholds=rr_thresholds, confidence_pct=fusion_pct,
        description="Fused Piezo and BMI270 respiratory motion while still.",
    )

    ibi = summary.get("aggregate_interbeat_metrics", {})
    resp_var = summary.get("aggregate_respiratory_variability", {})
    coupling = summary.get("aggregate_cardiorespiratory_coupling", {})
    ibi_base = _derived_interval_base(heart_base, multiplier=60000.0, unit="ms")
    breath_base = _derived_interval_base(resp_base, multiplier=60.0, unit="s")
    add("mean_ibi_ms", "Mean beat interval", "pulse_variability", ibi.get("mean_ibi_ms"), "ms", 0, base=ibi_base)
    add("ibi_sd_ms", "Beat interval SD", "pulse_variability", ibi.get("ibi_sd_ms"), "ms", 1,
        description="Mechanical SDNN-like variation; not validated ECG SDNN.")
    add("ibi_rmssd_ms", "Mechanical RMSSD", "pulse_variability", ibi.get("ibi_rmssd_ms"), "ms", 1)
    add("ibi_pnn50_pct", "Mechanical pNN50", "pulse_variability", ibi.get("ibi_pnn50_pct"), "%", 1)
    add("ibi_cv", "Beat interval CV", "pulse_variability", ibi.get("ibi_cv"), "ratio", 3)
    add("ibi_poincare_sd1_ms", "Poincare SD1", "pulse_variability", ibi.get("ibi_poincare_sd1_ms"), "ms", 1)
    add("ibi_poincare_sd2_ms", "Poincare SD2", "pulse_variability", ibi.get("ibi_poincare_sd2_ms"), "ms", 1)

    add("mean_breath_interval_s", "Mean breath interval", "respiratory_variability",
        resp_var.get("mean_breath_interval_s"), "s", 2, base=breath_base)
    add("breath_interval_sd_s", "Breath interval SD", "respiratory_variability",
        resp_var.get("breath_interval_sd_s"), "s", 2)
    add("breath_interval_rmssd_s", "Breath interval RMSSD", "respiratory_variability",
        resp_var.get("breath_interval_rmssd_s"), "s", 2)
    add("breath_interval_cv", "Breath interval CV", "respiratory_variability",
        resp_var.get("breath_interval_cv"), "ratio", 3)
    add("irregular_breathing_index", "Irregular breathing index", "respiratory_variability",
        resp_var.get("irregular_breathing_index"), "ratio", 3)

    add("beats_per_breath", "Beats per breath", "coupling", coupling.get("beats_per_breath"), "ratio", 2)
    add("respiratory_phase_locking_value", "Respiratory phase locking", "coupling",
        coupling.get("respiratory_phase_locking_value"), "0–1", 3)

    add("sleep_candidate_minutes", "Sleep-candidate time", "sleep", summary.get("sleep_candidate_minutes"), "min", 1,
        thresholds=[_threshold("at_or_above", config.sleep_minimum_minutes, "none", "minimum immobility bout")],
        description="Sustained immobility candidate, not a sleep stage.")
    add("sleep_candidate_efficiency_pct", "Sleep-candidate efficiency", "sleep",
        summary.get("sleep_candidate_efficiency_pct"), "%", 1)
    add("sleep_bout_count", "Sleep-candidate bouts", "sleep", summary.get("sleep_bout_count"), "bouts", 0)
    add("longest_sleep_candidate_bout_minutes", "Longest sleep-candidate bout", "sleep",
        summary.get("longest_sleep_candidate_bout_minutes"), "min", 1)
    add("sleep_fragmentation_events", "Sleep fragmentation", "sleep",
        summary.get("sleep_fragmentation_events"), "events", 0)
    add("sleep_fragmentation_events_per_hour", "Sleep fragmentation rate", "sleep",
        summary.get("sleep_fragmentation_events_per_hour"), "events/hour", 1)
    add("quiet_rest_minutes", "Quiet awake rest", "sleep", summary.get("quiet_rest_minutes"), "min", 1)

    add("active_minutes", "Active time", "activity", summary.get("active_minutes"), "min", 1)
    add("activity_enmo_mg", "Activity ENMO", "activity", _nested(summary, "activity_enmo_mg", "median"), "mg", 1,
        base=_engineering_base(config.active_enmo_mg, None, "active threshold"),
        thresholds=[_threshold("at_or_above", config.active_enmo_mg, "none", "classified as active")])
    add("moving_bout_count", "Movement bouts", "activity", summary.get("moving_bout_count"), "bouts", 0)
    add("position_change_count", "Position changes", "posture", summary.get("position_change_count"), "changes", 0,
        thresholds=[_threshold("angle_at_or_above", config.posture_change_degrees, "none", "degrees between gravity vectors")])
    add("orientation_transition_count", "Orientation transitions", "posture",
        summary.get("orientation_transition_count"), "transitions", 0)
    latest_orientation = windows[-1].orientation_bin if windows else None
    latest_context = windows[-1].context if windows else None
    add("current_orientation", "Current orientation bin", "posture", latest_orientation, "", 0,
        state="observed", description="Axis bin requires mounting calibration before naming posture.")
    add("current_context", "Current context", "activity", latest_context, "", 0, state="observed")

    pulse_amplitude = _median(window.pulse_amplitude for window in source_windows)
    respiratory_amplitude = _median(window.respiratory_amplitude for window in source_windows)
    add("pulse_amplitude", "Relative pulse amplitude", "signal", pulse_amplitude, "sensor units", 3)
    add("respiratory_amplitude", "Relative respiratory amplitude", "signal", respiratory_amplitude, "sensor units", 3)
    add("fusion_quality_pct", "Signal confidence", "signal", fusion_pct, "%", 0,
        base=_engineering_range(50.0, 100.0, "minimum accepted fusion quality"),
        thresholds=[_threshold("below", config.minimum_fusion_quality * 100.0, "info", "quality gate")],
        state=_quality_state(fusion_pct))
    coverage = _finite(summary.get("usable_data_coverage_pct"))
    add("usable_data_coverage_pct", "Usable data coverage", "signal", coverage, "%", 1,
        base=_engineering_range(80.0, 100.0, "recommended session coverage"), state=_coverage_state(coverage))
    artifact_pct = _finite(summary.get("artifact_window_pct"))
    add("artifact_window_pct", "Motion-artifact burden", "signal", artifact_pct, "%", 1,
        base=_engineering_range(0.0, 10.0, "preferred artifact burden"), state=_burden_state(artifact_pct))
    contact_pct = _finite(summary.get("contact_loss_window_pct"))
    add("contact_loss_window_pct", "Contact-loss burden", "signal", contact_pct, "%", 1,
        base=_engineering_range(0.0, 5.0, "preferred contact-loss burden"), state=_burden_state(contact_pct, fair=5.0))
    add("valid_window_count", "Valid analysis windows", "signal", summary.get("valid_window_count"), "windows", 0)
    add("possible_resp_pause_count", "Respiratory dropout candidates", "signal",
        sum(int(window.signal_metrics.get("possible_resp_pause_count") or 0) for window in windows), "events", 0,
        description="Could be breathing pause or sensor contact loss; not an apnea diagnosis.")
    impulse_pct = _finite(summary.get("piezo_impulse_artifact_window_pct"))
    add("piezo_impulse_artifact_window_pct", "Piezo press/impact artifact", "signal", impulse_pct, "%", 1,
        base=_engineering_range(0.0, 0.0, "no press/impact artifact preferred"),
        state=_burden_state(impulse_pct, fair=0.0),
        description="Large Piezo transient candidate; affected windows are not reported as heart-rate spikes.")
    motion_pct = _finite(summary.get("motion_artifact_window_pct"))
    add("motion_artifact_window_pct", "Dog-motion artifact", "signal", motion_pct, "%", 1,
        base=_engineering_range(0.0, 10.0, "preferred motion-artifact burden"),
        state=_burden_state(motion_pct))

    recovery = _recovery_payload(summary.get("recovery_events", []))
    latest_recovery = recovery.get("latest") or {}
    add("recovery_first_heart_rate_bpm", "Recovery starting HR", "recovery",
        latest_recovery.get("first_resting_heart_rate_bpm"), "bpm", 0)
    add("recovery_first_respiratory_rate_bpm", "Recovery starting RR", "recovery",
        latest_recovery.get("first_resting_respiratory_rate_bpm"), "breaths/min", 1)
    recovery_measurements = latest_recovery.get("measurements", {})
    for minute in (1, 3, 5, 10):
        measurement = recovery_measurements.get(f"minute_{minute}", {})
        add(f"heart_rate_recovery_{minute}m_bpm", f"HR recovery at {minute} min", "recovery",
            measurement.get("heart_rate_drop_from_first_bpm"), "bpm drop", 1,
            description="Positive means heart rate fell from the first resting recovery reading.")
        add(f"respiratory_rate_recovery_{minute}m_bpm", f"RR recovery at {minute} min", "recovery",
            measurement.get("respiratory_rate_drop_from_first_bpm"), "breaths/min drop", 1,
            description="Positive means respiratory rate fell from the first resting recovery reading.")
    groups = {
        category: [metric_id for metric_id, metric in metrics.items() if metric["category"] == category]
        for category in (
            "vitals", "pulse_variability", "respiratory_variability", "coupling",
            "sleep", "activity", "posture", "recovery", "signal",
        )
    }
    highest_severity = max(
        (str(alert.get("severity", "none")) for alert in alerts),
        key=lambda item: _SEVERITY_RANK.get(item, 0),
        default="none",
    )
    return {
        "schema_version": "vitalsense.frontend.v1",
        "dog_id": profile.dog_id,
        "measurement_context": "sleep_candidate_or_quiet_rest",
        "reference_profile": reference["profile"],
        "baseline_training": {
            "available": bool(baseline_score.get("available")),
            "eligible": bool(baseline_score.get("eligible")),
            "minimum_days": 7,
            "minimum_eligible_samples": 100,
            "context": baseline_score.get("context", "sleep_candidate"),
        },
        "overall": {
            "highest_alert_severity": highest_severity,
            "alert_count": len(alerts),
            "personal_anomaly_score_0_1": baseline_score.get("overall_anomaly_score"),
            "usable_data_coverage_pct": coverage,
        },
        "metrics": metrics,
        "groups": groups,
        "recovery": recovery,
        "alerts": alerts,
        "status_legend": {
            "severity": ["none", "info", "watch", "high", "urgent", "unavailable"],
            "state": [
                "within_reference", "below_reference", "above_reference", "not_established",
                "good", "fair", "poor", "observed", "unavailable",
            ],
        },
        "disclaimer": "Research/trend screening only. Population ranges are descriptive, not diagnostic limits.",
    }


def _reference_for_profile(profile: DogProfile) -> dict[str, Any]:
    age = profile.age_years
    weight = profile.weight_kg
    source = "AI-COLLAR 2025, apparently healthy dogs, resting immobile; median and IQR"
    assumption: str | None = None
    if age is not None and age <= 1.0:
        label = "puppy (<=12 months)"
        heart = _population_base(78.5, 69.1, 96.8, source, label)
        resp = _population_base(20.1, 16.0, 25.7, source, label)
    elif age is not None and age >= 10.0:
        label = "senior (>=10 years)"
        heart = _population_base(68.7, 60.8, 77.2, source, label)
        resp = _population_base(15.7, 13.1, 18.7, source, "older than 12 months")
    elif age is not None and weight is not None and weight <= 10.0:
        label = "adult small dog (<=10 kg)"
        heart = _population_base(65.0, 60.9, 69.4, source, label)
        resp = _population_base(17.2, 15.0, 19.9, source, label)
    elif age is not None and weight is not None and weight > 10.0:
        label = "adult medium/large dog (>10 kg)"
        heart = _population_base(59.5, 54.6, 64.2, source, label)
        resp = _population_base(15.8, 13.7, 18.3, source, label)
    elif age is not None:
        label = "adult dog (weight not supplied)"
        heart = _population_base(60.5, 55.2, 65.3, source, label)
        resp = _population_base(15.7, 13.1, 18.7, source, label)
        assumption = "Adult reference used because weight was not supplied."
    else:
        label = "adult fallback (age/weight not supplied)"
        heart = _population_base(60.5, 55.2, 65.3, source, label)
        resp = _population_base(15.7, 13.1, 18.7, source, label)
        assumption = "Adult reference is a UI fallback until age and weight are supplied."
    return {
        "profile": {
            "label": label,
            "age_years": profile.age_years,
            "weight_kg": profile.weight_kg,
            "breed": profile.breed,
            "assumption": assumption,
        },
        "heart_rate_bpm": heart,
        "respiratory_rate_bpm": resp,
    }


def _baseline_for_metric(
    metric: str,
    baseline_score: dict[str, Any],
    population: dict[str, Any],
) -> dict[str, Any]:
    score = baseline_score.get("metrics", {}).get(metric, {})
    if baseline_score.get("eligible") and score.get("baseline_median") is not None:
        center = float(score["baseline_median"])
        scale = float(score.get("baseline_mad_scale") or 0.0)
        return {
            "kind": "personal_robust_band",
            "center": center,
            "lower": max(0.0, center - 3.0 * scale),
            "upper": center + 3.0 * scale,
            "statistic": "personal median +/- 3 robust MAD scales",
            "context": baseline_score.get("context", "sleep_candidate"),
            "source": "this dog's eligible baseline",
            "personalized": True,
        }
    return population


def _population_base(center: float, lower: float, upper: float, source: str, subgroup: str) -> dict[str, Any]:
    return {
        "kind": "population_iqr",
        "center": center,
        "lower": lower,
        "upper": upper,
        "statistic": "median and interquartile range",
        "context": "resting immobile",
        "source": source,
        "subgroup": subgroup,
        "personalized": False,
    }


def _unestablished_base() -> dict[str, Any]:
    return {
        "kind": "personal_baseline_required",
        "center": None,
        "lower": None,
        "upper": None,
        "statistic": None,
        "context": None,
        "source": "No validated universal canine reference interval",
        "personalized": False,
    }


def _engineering_base(lower: float | None, upper: float | None, label: str) -> dict[str, Any]:
    return {
        "kind": "engineering_setting",
        "center": None,
        "lower": lower,
        "upper": upper,
        "statistic": label,
        "context": "device processing",
        "source": "VitalSense default configuration; validate on final hardware",
        "personalized": False,
    }


def _engineering_range(lower: float, upper: float, label: str) -> dict[str, Any]:
    return _engineering_base(lower, upper, label)


def _derived_interval_base(rate_base: dict[str, Any], multiplier: float, unit: str) -> dict[str, Any]:
    lower_rate = _finite(rate_base.get("lower"))
    upper_rate = _finite(rate_base.get("upper"))
    center_rate = _finite(rate_base.get("center"))
    if lower_rate is None or upper_rate is None or center_rate is None or lower_rate <= 0:
        return _unestablished_base()
    return {
        "kind": f"derived_from_{rate_base.get('kind', 'rate_reference')}",
        "center": multiplier / center_rate,
        "lower": multiplier / upper_rate,
        "upper": multiplier / lower_rate,
        "statistic": f"inverse of rate reference, expressed in {unit}",
        "context": rate_base.get("context"),
        "source": rate_base.get("source"),
        "personalized": bool(rate_base.get("personalized")),
    }


def _recovery_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "available": False,
            "event_count": 0,
            "latest": None,
            "base": _unestablished_base(),
            "message": "No qualifying >=2-minute activity-to-rest transition was observed.",
        }
    latest = events[-1]
    return {
        "available": True,
        "event_count": len(events),
        "latest": latest,
        "base": _unestablished_base(),
        "message": "Interpret against this dog's repeated recovery history; no universal canine cutoff is applied.",
    }


def _metric_alert_severity(metric_id: str, alerts: list[dict[str, Any]]) -> str:
    aliases = {
        "heart_rate_bpm": {"heart_rate", "heart_rate_bpm"},
        "respiratory_rate_bpm": {"respiratory_rate", "respiratory_rate_bpm"},
        "ibi_cv": {"inter_beat_variability", "mechanical_interbeat_variability"},
        "possible_resp_pause_count": {"respiratory_motion"},
    }
    names = aliases.get(metric_id, {metric_id})
    def applies(alert: dict[str, Any]) -> bool:
        if alert.get("metric") in names:
            return True
        rule_id = str(alert.get("rule_id", ""))
        if metric_id == "heart_rate_bpm" and rule_id.startswith("HR_"):
            return True
        if metric_id == "respiratory_rate_bpm" and rule_id.startswith("RR_"):
            return True
        if metric_id == "ibi_cv" and rule_id.startswith("IRREGULAR_PULSE"):
            return True
        if metric_id == "possible_resp_pause_count" and rule_id.startswith("RESP_PAUSE"):
            return True
        return False

    severities = [str(alert.get("severity", "none")) for alert in alerts if applies(alert)]
    return max(severities, key=lambda item: _SEVERITY_RANK.get(item, 0), default="none")


def _reference_state(value: float | None, base: dict[str, Any]) -> str:
    if value is None:
        return "unavailable"
    lower = _finite(base.get("lower"))
    upper = _finite(base.get("upper"))
    if lower is None or upper is None:
        return "not_established"
    if value < lower:
        return "below_reference"
    if value > upper:
        return "above_reference"
    return "within_reference"


def _quality_state(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value >= 75.0:
        return "good"
    if value >= 50.0:
        return "fair"
    return "poor"


def _coverage_state(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value >= 80.0:
        return "good"
    if value >= 50.0:
        return "fair"
    return "poor"


def _burden_state(value: float | None, fair: float = 10.0) -> str:
    if value is None:
        return "unavailable"
    if value <= fair:
        return "good"
    if value <= 25.0:
        return "fair"
    return "poor"


def _threshold(operator: str, value: float, severity: str, persistence: str) -> dict[str, Any]:
    return {"operator": operator, "value": value, "severity": severity, "persistence": persistence}


def _nested(payload: dict[str, Any], first: str, second: str) -> Any:
    block = payload.get(first, {})
    return block.get(second) if isinstance(block, dict) else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [_finite(value) for value in values]
    clean = [value for value in finite if value is not None]
    return float(np.median(clean)) if clean else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _percent(value: Any) -> float | None:
    numeric = _finite(value)
    return numeric * 100.0 if numeric is not None else None


def _round_or_none(value: float | None, decimals: int) -> float | None:
    return round(value, decimals) if value is not None else None


def _display(value: float | int | str | None, unit: str, decimals: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    formatted = f"{float(value):.{decimals}f}"
    return f"{formatted} {unit}".strip()
