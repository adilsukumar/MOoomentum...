"""Conservative, auditable alert rules for research use."""

from __future__ import annotations

from typing import Any, Callable

from .models import WindowResult


def evaluate_alerts(
    windows: list[WindowResult],
    baseline_score: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    eligible = [
        window
        for window in windows
        if window.valid and window.context in {"sleep_candidate", "quiet_awake_rest"}
    ]
    alerts: list[dict[str, Any]] = []

    _append_if_sustained(
        alerts,
        eligible,
        "RR_SLEEP_25",
        "watch",
        lambda window: window.context == "sleep_candidate"
        and window.respiratory_rate_bpm is not None
        and window.respiratory_rate_bpm > 25.0,
        3,
        "Sleeping respiratory rate was repeatedly above 25 breaths/min. Confirm manually and monitor the personal multi-day trend.",
        evaluation_span=5,
    )
    _append_if_sustained(
        alerts,
        eligible,
        "RR_SLEEP_30",
        "high",
        lambda window: window.respiratory_rate_bpm is not None and window.respiratory_rate_bpm >= 30.0,
        2,
        "Sleeping/resting respiratory rate was repeatedly at or above 30 breaths/min. Confirm for a full minute and contact a veterinarian if confirmed or symptoms are present.",
        evaluation_span=3,
    )
    _append_if_sustained(
        alerts,
        eligible,
        "RR_REST_40",
        "urgent",
        lambda window: window.respiratory_rate_bpm is not None and window.respiratory_rate_bpm >= 40.0,
        9,
        "Resting respiratory rate was repeatedly at or above 40 breaths/min. Seek urgent veterinary advice, especially with visible breathing effort.",
        evaluation_span=9,
    )
    _append_if_sustained(
        alerts,
        eligible,
        "RR_LOW",
        "info",
        lambda window: window.respiratory_rate_bpm is not None and window.respiratory_rate_bpm < 8.0,
        2,
        "Very low respiratory motion was detected. Check collar contact and visually observe breathing; this is not an apnea diagnosis.",
        evaluation_span=3,
    )
    _append_if_sustained(
        alerts,
        eligible,
        "HR_LOW_40",
        "high",
        lambda window: window.heart_rate_bpm is not None and window.heart_rate_bpm < 40.0,
        3,
        "Mechanical pulse rate was repeatedly below 40 bpm at rest. Confirm manually and contact a veterinarian if confirmed or symptomatic.",
        evaluation_span=5,
    )
    _append_if_sustained(
        alerts,
        eligible,
        "HR_LOW_30",
        "urgent",
        lambda window: window.heart_rate_bpm is not None and window.heart_rate_bpm < 30.0,
        1,
        "Mechanical pulse rate was below 30 bpm. Seek urgent veterinary advice if confirmed; ECG is required for rhythm diagnosis.",
    )
    _append_if_sustained(
        alerts,
        eligible,
        "HR_HIGH_140",
        "high",
        lambda window: window.heart_rate_bpm is not None and window.heart_rate_bpm > 140.0,
        3,
        "Mechanical pulse rate was repeatedly above 140 bpm at rest. Check heat, pain, fear and illness; confirm and contact a veterinarian if persistent.",
        evaluation_span=5,
    )
    _append_if_sustained(
        alerts,
        eligible,
        "HR_HIGH_180",
        "urgent",
        lambda window: window.heart_rate_bpm is not None and window.heart_rate_bpm > 180.0,
        1,
        "Mechanical pulse rate exceeded 180 bpm at rest. Seek urgent veterinary advice if confirmed or symptoms are present.",
    )

    if baseline_score and baseline_score.get("eligible"):
        for metric, score in baseline_score.get("metrics", {}).items():
            if abs(float(score.get("relative_change_pct", 0.0))) >= 20.0:
                alerts.append(
                    {
                        "rule_id": "PERSONAL_BASELINE_SHIFT",
                        "severity": "watch",
                        "metric": metric,
                        "observed": score.get("current"),
                        "message": f"{metric} shifted at least 20% from this dog's eligible personal baseline. Confirm the trend and review context.",
                    }
                )

    irregular_windows = [
        window
        for window in eligible
        if (window.interbeat_metrics.get("ibi_cv") or 0.0) > 0.25 and window.pulse_count >= 20
    ]
    if len(irregular_windows) >= 3:
        alerts.append(
            {
                "rule_id": "IRREGULAR_PULSE_CANDIDATE",
                "severity": "watch",
                "metric": "mechanical_interbeat_variability",
                "observed": len(irregular_windows),
                "message": "Repeated mechanical pulse irregularity was detected. This is not an arrhythmia diagnosis; ECG/Holter confirmation is required.",
            }
        )
    pause_windows = [
        window
        for window in eligible
        if float(window.signal_metrics.get("possible_resp_pause_count") or 0.0) > 0.0
    ]
    if len(pause_windows) >= 2:
        alerts.append(
            {
                "rule_id": "RESP_PAUSE_OR_SIGNAL_LOSS_CANDIDATE",
                "severity": "high",
                "metric": "respiratory_motion",
                "observed": len(pause_windows),
                "message": "Repeated simultaneous respiratory-amplitude dropouts were detected. Check collar contact and observe the dog; this is not a sleep-apnea diagnosis.",
            }
        )
    return alerts


def _append_if_sustained(
    alerts: list[dict[str, Any]],
    windows: list[WindowResult],
    rule_id: str,
    severity: str,
    predicate: Callable[[WindowResult], bool],
    required: int,
    message: str,
    evaluation_span: int | None = None,
) -> None:
    matches = [window for window in windows if predicate(window)]
    span = evaluation_span or required
    qualifying_group: list[WindowResult] | None = None
    for start in range(max(len(windows) - span + 1, 0)):
        group = windows[start : start + span]
        group_matches = [window for window in group if predicate(window)]
        if len(group_matches) >= required:
            qualifying_group = group_matches
            break
    if qualifying_group is None:
        return
    alerts.append(
        {
            "rule_id": rule_id,
            "severity": severity,
            "matching_window_count": len(matches),
            "first_match_s": qualifying_group[0].start_s,
            "last_match_s": qualifying_group[-1].end_s,
            "message": message,
        }
    )
