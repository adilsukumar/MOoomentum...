"""Session aggregation, sleep/rest bouts, recovery, and personal baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .metrics import percentile_summary
from .models import AnalysisConfig, WindowResult
from .signal import angle_degrees, clamp


@dataclass(slots=True)
class BaselineMetric:
    median: float
    mad_scale: float
    sample_count: int
    day_count: int


@dataclass(slots=True)
class PersonalBaseline:
    dog_id: str
    context: str
    metrics: dict[str, BaselineMetric]
    baseline_days: int
    eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dog_id": self.dog_id,
            "context": self.context,
            "baseline_days": self.baseline_days,
            "eligible": self.eligible,
            "metrics": {key: asdict(value) for key, value in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersonalBaseline":
        return cls(
            dog_id=str(payload.get("dog_id", "unknown")),
            context=str(payload.get("context", "sleep_candidate")),
            baseline_days=int(payload.get("baseline_days", 0)),
            eligible=bool(payload.get("eligible", False)),
            metrics={key: BaselineMetric(**value) for key, value in payload.get("metrics", {}).items()},
        )


def assign_sustained_contexts(windows: list[WindowResult], config: AnalysisConfig) -> None:
    minimum_seconds = config.sleep_minimum_minutes * 60.0
    for window in windows:
        if window.panting_candidate:
            window.context = "panting_or_unusable"
        elif window.static_candidate:
            window.context = "quiet_awake_rest"
        else:
            window.context = "moving"

    # Reserve the first ten minutes after a >=2 minute activity bout for recovery,
    # preventing recovery tachycardia/tachypnea from being treated as sleeping vitals.
    minimum_active_windows = max(2, int(120.0 / config.step_seconds))
    for index in range(minimum_active_windows, len(windows)):
        preceding = windows[index - minimum_active_windows : index]
        if not all(window.context == "moving" for window in preceding):
            continue
        if windows[index].context != "quiet_awake_rest":
            continue
        recovery_start = windows[index].start_s
        for current in windows[index:]:
            if current.start_s - recovery_start > 10.0 * 60.0 or not current.static_candidate or current.panting_candidate:
                break
            current.context = "post_activity_recovery"

    index = 0
    while index < len(windows):
        if windows[index].context != "quiet_awake_rest":
            index += 1
            continue
        start = index
        while index + 1 < len(windows) and windows[index + 1].context == "quiet_awake_rest":
            index += 1
        stop = index
        duration = windows[stop].end_s - windows[start].start_s
        if duration >= minimum_seconds:
            for current in range(start, stop + 1):
                windows[current].context = "sleep_candidate"
        index += 1


def summarize_session(windows: list[WindowResult], config: AnalysisConfig) -> dict[str, Any]:
    if not windows:
        return {"window_count": 0, "valid_window_count": 0, "usable_data_coverage_pct": 0.0}
    assign_sustained_contexts(windows, config)

    valid = [window for window in windows if window.valid]
    duration_s = max(window.end_s for window in windows) - min(window.start_s for window in windows)
    step = config.step_seconds
    context_seconds: dict[str, float] = {}
    for window in windows:
        context_seconds[window.context] = context_seconds.get(window.context, 0.0) + step

    position_changes = 0
    orientation_transitions = 0
    for previous, current in zip(windows, windows[1:]):
        if previous.orientation_bin != current.orientation_bin:
            orientation_transitions += 1
        if angle_degrees(np.asarray(previous.gravity_vector_g), np.asarray(current.gravity_vector_g)) >= config.posture_change_degrees:
            position_changes += 1

    sleep_bouts = _bouts(windows, "sleep_candidate")
    moving_bouts = _bouts(windows, "moving")
    awakenings = 0
    for previous, current in zip(windows, windows[1:]):
        if previous.context == "sleep_candidate" and current.context == "moving":
            awakenings += 1

    sleep_hours = context_seconds.get("sleep_candidate", 0.0) / 3600.0
    heart_values = [window.heart_rate_bpm for window in valid if window.heart_rate_bpm is not None]
    resp_values = [window.respiratory_rate_bpm for window in valid if window.respiratory_rate_bpm is not None]
    fusion_values = [window.fusion_quality for window in windows]
    enmo_values = [window.activity_enmo_mg for window in windows]

    summary: dict[str, Any] = {
        "duration_seconds": float(duration_s),
        "window_count": len(windows),
        "valid_window_count": len(valid),
        "usable_data_coverage_pct": float(100.0 * len(valid) / max(len(windows), 1)),
        "context_minutes": {key: float(value / 60.0) for key, value in context_seconds.items()},
        "sleep_candidate_minutes": float(context_seconds.get("sleep_candidate", 0.0) / 60.0),
        "quiet_rest_minutes": float(context_seconds.get("quiet_awake_rest", 0.0) / 60.0),
        "active_minutes": float(context_seconds.get("moving", 0.0) / 60.0),
        "sleep_candidate_efficiency_pct": float(
            100.0 * context_seconds.get("sleep_candidate", 0.0) / max(duration_s, 1.0)
        ),
        "sleep_bout_count": len(sleep_bouts),
        "longest_sleep_candidate_bout_minutes": float(max((bout[1] - bout[0]) for bout in sleep_bouts) / 60.0)
        if sleep_bouts
        else 0.0,
        "sleep_fragmentation_events": awakenings,
        "sleep_fragmentation_events_per_hour": float(awakenings / max(sleep_hours, 1e-9)) if sleep_hours else 0.0,
        "moving_bout_count": len(moving_bouts),
        "position_change_count": position_changes,
        "orientation_transition_count": orientation_transitions,
        "heart_rate_bpm": percentile_summary(heart_values),
        "respiratory_rate_bpm": percentile_summary(resp_values),
        "fusion_quality": percentile_summary(fusion_values),
        "activity_enmo_mg": percentile_summary(enmo_values),
        "artifact_window_pct": float(
            100.0
            * sum(
                "motion_artifact" in window.reason_codes
                or "piezo_impulse_artifact_candidate" in window.reason_codes
                or "piezo_clipping_or_flatline" in window.reason_codes
                for window in windows
            )
            / len(windows)
        ),
        "motion_artifact_window_pct": float(
            100.0 * sum("motion_artifact" in window.reason_codes for window in windows) / len(windows)
        ),
        "piezo_impulse_artifact_window_pct": float(
            100.0
            * sum("piezo_impulse_artifact_candidate" in window.reason_codes for window in windows)
            / len(windows)
        ),
        "contact_loss_window_pct": float(
            100.0 * sum("piezo_contact_loss" in window.reason_codes for window in windows) / len(windows)
        ),
    }
    summary["recovery_events"] = detect_recovery_events(windows, config)
    summary["aggregate_interbeat_metrics"] = aggregate_metric_dicts(
        [window.interbeat_metrics for window in valid if window.interbeat_metrics]
    )
    summary["aggregate_respiratory_variability"] = aggregate_metric_dicts(
        [window.respiratory_variability for window in valid if window.respiratory_variability]
    )
    summary["aggregate_cardiorespiratory_coupling"] = aggregate_metric_dicts(
        [window.cardiorespiratory_coupling for window in valid if window.cardiorespiratory_coupling]
    )
    return summary


def detect_recovery_events(windows: list[WindowResult], config: AnalysisConfig) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    minimum_active_windows = max(2, int(120.0 / config.step_seconds))
    index = minimum_active_windows
    while index < len(windows):
        preceding = windows[index - minimum_active_windows : index]
        if all(window.context == "moving" for window in preceding) and windows[index].static_candidate:
            start_time = windows[index].start_s
            measurements: dict[str, Any] = {}
            first_hr = next(
                (window.heart_rate_bpm for window in windows[index:] if window.valid and window.heart_rate_bpm is not None),
                None,
            )
            first_rr = next(
                (
                    window.respiratory_rate_bpm
                    for window in windows[index:]
                    if window.valid and window.respiratory_rate_bpm is not None
                ),
                None,
            )
            for minute in (1, 3, 5, 10):
                target = start_time + minute * 60.0
                nearest = min(windows[index:], key=lambda window: abs(window.start_s - target), default=None)
                if nearest and abs(nearest.start_s - target) <= config.step_seconds:
                    measurements[f"minute_{minute}"] = {
                        "heart_rate_bpm": nearest.heart_rate_bpm,
                        "respiratory_rate_bpm": nearest.respiratory_rate_bpm,
                        "heart_rate_drop_from_first_bpm": (
                            float(first_hr - nearest.heart_rate_bpm)
                            if first_hr is not None and nearest.heart_rate_bpm is not None
                            else None
                        ),
                        "respiratory_rate_drop_from_first_bpm": (
                            float(first_rr - nearest.respiratory_rate_bpm)
                            if first_rr is not None and nearest.respiratory_rate_bpm is not None
                            else None
                        ),
                    }
            events.append(
                {
                    "recovery_start_s": float(start_time),
                    "preceding_active_minutes": float(minimum_active_windows * config.step_seconds / 60.0),
                    "first_resting_heart_rate_bpm": first_hr,
                    "first_resting_respiratory_rate_bpm": first_rr,
                    "measurements": measurements,
                }
            )
            index += minimum_active_windows
        else:
            index += 1
    return events


def build_personal_baseline(
    records: Iterable[dict[str, Any]],
    dog_id: str,
    context: str = "sleep_candidate",
    minimum_days: int = 7,
    minimum_samples: int = 100,
) -> PersonalBaseline:
    rows = list(records)
    days = {str(row.get("date", row.get("session_id", index))) for index, row in enumerate(rows)}
    metric_names = ("heart_rate_bpm", "respiratory_rate_bpm")
    metrics: dict[str, BaselineMetric] = {}
    for name in metric_names:
        values = np.asarray(
            [float(row[name]) for row in rows if row.get("context", context) == context and row.get(name) is not None],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        median = float(np.median(values))
        mad_scale = float(1.4826 * np.median(np.abs(values - median)))
        minimum_scale = 1.0 if name == "heart_rate_bpm" else 0.5
        metrics[name] = BaselineMetric(
            median=median,
            mad_scale=max(mad_scale, minimum_scale),
            sample_count=len(values),
            day_count=len(days),
        )
    eligible = len(days) >= minimum_days and all(
        metric.sample_count >= minimum_samples for metric in metrics.values()
    ) and bool(metrics)
    return PersonalBaseline(dog_id=dog_id, context=context, metrics=metrics, baseline_days=len(days), eligible=eligible)


def evaluate_multiday_trends(
    records: Iterable[dict[str, Any]],
    baseline: PersonalBaseline,
    context: str | None = None,
) -> dict[str, Any]:
    """Calculate daily and rolling three-day medians plus persistent 20% shifts."""
    selected_context = context or baseline.context
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in records:
        if row.get("context", selected_context) != selected_context:
            continue
        date = str(row.get("date", "unknown"))
        day = grouped.setdefault(date, {name: [] for name in baseline.metrics})
        for name in baseline.metrics:
            value = row.get(name)
            if value is not None and np.isfinite(float(value)):
                day[name].append(float(value))

    daily: list[dict[str, Any]] = []
    for date in sorted(grouped):
        daily.append(
            {
                "date": date,
                **{
                    name: float(np.median(values)) if values else None
                    for name, values in grouped[date].items()
                },
            }
        )

    points: list[dict[str, Any]] = []
    for index in range(2, len(daily)):
        point: dict[str, Any] = {"date": daily[index]["date"], "metrics": {}}
        for name, model in baseline.metrics.items():
            recent = [daily[position].get(name) for position in range(index - 2, index + 1)]
            recent = [float(value) for value in recent if value is not None]
            if len(recent) < 3:
                continue
            rolling = float(np.median(recent))
            relative = float((rolling - model.median) / max(model.median, 1e-9))
            point["metrics"][name] = {
                "rolling_3day_median": rolling,
                "relative_change_pct": relative * 100.0,
                "robust_z": float((rolling - model.median) / max(model.mad_scale, 1e-9)),
            }
        points.append(point)

    alerts: list[dict[str, Any]] = []
    for name in baseline.metrics:
        recent_scores = [
            point["metrics"][name]["relative_change_pct"]
            for point in points[-3:]
            if name in point["metrics"]
        ]
        if len(recent_scores) == 3 and all(abs(score) >= 20.0 for score in recent_scores):
            directions = {1 if score > 0 else -1 for score in recent_scores}
            if len(directions) == 1:
                alerts.append(
                    {
                        "rule_id": "PERSISTENT_3DAY_BASELINE_SHIFT",
                        "severity": "watch",
                        "metric": name,
                        "direction": "high" if recent_scores[-1] > 0 else "low",
                        "latest_relative_change_pct": recent_scores[-1],
                        "message": f"The rolling three-day {name} median remained at least 20% away from baseline for three consecutive evaluations.",
                    }
                )
    return {
        "context": selected_context,
        "daily_medians": daily,
        "rolling_3day": points,
        "alerts": alerts,
    }


def score_against_baseline(summary: dict[str, Any], baseline: PersonalBaseline | None) -> dict[str, Any]:
    if baseline is None:
        return {"available": False, "eligible": False, "metrics": {}, "overall_anomaly_score": None}
    scores: dict[str, Any] = {}
    magnitudes: list[float] = []
    for name, model in baseline.metrics.items():
        current_block = summary.get(name, {})
        current = current_block.get("median") if isinstance(current_block, dict) else current_block
        if current is None:
            continue
        robust_z = float((current - model.median) / max(model.mad_scale, 1e-9))
        relative_change = float((current - model.median) / max(model.median, 1e-9))
        anomaly = clamp(abs(robust_z) / 6.0)
        scores[name] = {
            "current": current,
            "baseline_median": model.median,
            "baseline_mad_scale": model.mad_scale,
            "robust_z": robust_z,
            "relative_change_pct": relative_change * 100.0,
            "anomaly_score_0_1": anomaly,
        }
        magnitudes.append(anomaly)
    return {
        "available": True,
        "eligible": baseline.eligible,
        "context": baseline.context,
        "metrics": scores,
        "overall_anomaly_score": float(max(magnitudes)) if magnitudes else None,
    }


def aggregate_metric_dicts(metrics: list[dict[str, float | None]]) -> dict[str, float | None]:
    names = sorted({name for metric in metrics for name in metric})
    result: dict[str, float | None] = {}
    for name in names:
        values = [float(metric[name]) for metric in metrics if metric.get(name) is not None]
        result[name] = float(np.median(values)) if values else None
    return result


def _bouts(windows: list[WindowResult], context: str) -> list[tuple[float, float]]:
    bouts: list[tuple[float, float]] = []
    start: float | None = None
    end: float | None = None
    for window in windows:
        if window.context == context:
            start = window.start_s if start is None else start
            end = window.end_s
        elif start is not None and end is not None:
            bouts.append((start, end))
            start = end = None
    if start is not None and end is not None:
        bouts.append((start, end))
    return bouts
