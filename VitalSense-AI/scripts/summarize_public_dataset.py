"""Summarize the open Invoxia Dog Health Vitals CSV without modifying it."""

from __future__ import annotations

import ast
import csv
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "public" / "invoxia_dog_health_vitals" / "dataset.csv"


def parse_list(value: str):
    return ast.literal_eval(value) if value else []


def main() -> None:
    with DATASET.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    dogs = {}
    hr_values: list[float] = []
    br_values: list[float] = []
    pulse_count = 0
    bad_ecg_seconds = 0.0
    missing_ecg_paths: list[str] = []
    sessions_without_ecg_path = 0

    for row in rows:
        dogs[row["pet_id"]] = {
            "breed": row["breeds"],
            "weight": float(row["weight"]),
            "age": float(row["age"]),
        }
        hr_values.extend(float(s["value"]) for s in parse_list(row["segments_hr"]))
        br_values.extend(float(s["value"]) for s in parse_list(row["segments_br"]))
        pulse_count += len(parse_list(row["ecg_pulses"]))
        bad_ecg_seconds += sum(float(end) - float(start) for start, end in parse_list(row["bad_ecg"]))
        if not row["ecg_path"]:
            sessions_without_ecg_path += 1
        elif not (DATASET.parent / row["ecg_path"]).is_file():
            missing_ecg_paths.append(row["ecg_path"])

    total_seconds = sum(float(row["duration"]) for row in rows)
    breed_counts = Counter(dog["breed"] for dog in dogs.values())
    weights = [dog["weight"] for dog in dogs.values()]
    ages = [dog["age"] for dog in dogs.values()]

    print(f"sessions: {len(rows)}")
    print(f"dogs: {len(dogs)}")
    print(f"recording_hours: {total_seconds / 3600:.2f}")
    print(f"ecg_pulses: {pulse_count}")
    print(f"bad_ecg_hours (interval sum): {bad_ecg_seconds / 3600:.2f}")
    print(
        "missing_referenced_ecg_files: "
        f"{len(missing_ecg_paths)} session references / {len(set(missing_ecg_paths))} unique paths"
    )
    print(f"sessions_without_ecg_path: {sessions_without_ecg_path}")
    print(f"age_years: {min(ages):.2f}..{max(ages):.2f}; median={statistics.median(ages):.2f}")
    print(f"weight_kg: {min(weights):.2f}..{max(weights):.2f}; median={statistics.median(weights):.2f}")
    if hr_values:
        print(f"labeled_hr_bpm: {min(hr_values):.2f}..{max(hr_values):.2f}; median={statistics.median(hr_values):.2f}")
    if br_values:
        print(f"labeled_rr_bpm: {min(br_values):.2f}..{max(br_values):.2f}; median={statistics.median(br_values):.2f}")
    print("breeds:")
    for breed, count in sorted(breed_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {breed}: {count}")


if __name__ == "__main__":
    main()
