"""Command-line interface for VitalSense analysis and baseline generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import (
    load_baseline,
    load_config,
    load_json,
    load_profile,
    load_sensor_csv,
    save_sensor_csv,
    write_json,
)
from .longitudinal import build_personal_baseline, evaluate_multiday_trends
from .models import DogProfile
from .pipeline import analyze_session
from .synthetic import generate_synthetic_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vitalsense", description="VitalSense canine vital-sign research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a Piezo + BMI270 sensor CSV")
    analyze.add_argument("input_csv", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--profile", type=Path)
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--baseline", type=Path)

    demo = subparsers.add_parser("demo", help="Generate and analyze a deterministic synthetic session")
    demo.add_argument("--sensor-output", type=Path, required=True)
    demo.add_argument("--analysis-output", type=Path, required=True)
    demo.add_argument("--duration-minutes", type=float, default=20.0)
    demo.add_argument("--heart-rate", type=float, default=72.0)
    demo.add_argument("--respiratory-rate", type=float, default=15.0)

    baseline = subparsers.add_parser("baseline", help="Build a personal baseline from JSON/JSONL metric records")
    baseline.add_argument("history", type=Path)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--dog-id", required=True)
    baseline.add_argument("--context", default="sleep_candidate")
    baseline.add_argument("--minimum-days", type=int, default=7)
    baseline.add_argument("--minimum-samples", type=int, default=100)

    trend = subparsers.add_parser("trend", help="Evaluate multi-day records against a personal baseline")
    trend.add_argument("history", type=Path)
    trend.add_argument("--baseline", type=Path, required=True)
    trend.add_argument("--output", type=Path, required=True)
    trend.add_argument("--context")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        batch = load_sensor_csv(args.input_csv)
        result = analyze_session(
            batch,
            profile=load_profile(args.profile),
            config=load_config(args.config),
            baseline=load_baseline(args.baseline),
        )
        write_json(result, args.output)
        _print_summary(result)
        return 0

    if args.command == "demo":
        batch = generate_synthetic_session(
            duration_s=args.duration_minutes * 60.0,
            heart_rate_bpm=args.heart_rate,
            respiratory_rate_bpm=args.respiratory_rate,
        )
        save_sensor_csv(batch, args.sensor_output)
        result = analyze_session(
            batch,
            profile=DogProfile(dog_id="synthetic-dog", age_years=4.0, weight_kg=22.0, breed="synthetic"),
        )
        write_json(result, args.analysis_output)
        _print_summary(result)
        return 0

    if args.command == "baseline":
        records = _load_history(args.history)
        baseline = build_personal_baseline(
            records,
            dog_id=args.dog_id,
            context=args.context,
            minimum_days=args.minimum_days,
            minimum_samples=args.minimum_samples,
        )
        write_json(baseline.to_dict(), args.output)
        print(json.dumps(baseline.to_dict(), indent=2))
        return 0
    if args.command == "trend":
        baseline = load_baseline(args.baseline)
        if baseline is None:
            raise ValueError("A baseline file is required")
        result = evaluate_multiday_trends(_load_history(args.history), baseline, context=args.context)
        write_json(result, args.output)
        print(json.dumps({"days": len(result["daily_medians"]), "alerts": result["alerts"]}, indent=2))
        return 0
    return 2


def _load_history(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError("Baseline history must be a JSON list, {records: [...]}, or JSONL")


def _print_summary(result: dict) -> None:
    summary = result["summary"]
    compact = {
        "duration_seconds": summary.get("duration_seconds"),
        "usable_data_coverage_pct": summary.get("usable_data_coverage_pct"),
        "heart_rate_bpm": summary.get("heart_rate_bpm"),
        "respiratory_rate_bpm": summary.get("respiratory_rate_bpm"),
        "sleep_candidate_minutes": summary.get("sleep_candidate_minutes"),
        "active_minutes": summary.get("active_minutes"),
        "alert_count": len(result.get("alerts", [])),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
