from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class BreedEvidence:
    name: str
    odds_ratio_vs_labrador: float
    significantly_higher: bool
    evidence_note: str


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict[str, dict[str, object]], str]:
    path = files("smart_collar_temperature").joinpath(
        "config/breed_heat_evidence.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    note = str(data["study"]["population_note"])
    index: dict[str, dict[str, object]] = {}
    for row in data["breeds"]:
        names = [row["name"], *row.get("aliases", [])]
        for name in names:
            index[_normalize(str(name))] = row
    return index, note


def get_breed_evidence(breed: str) -> BreedEvidence | None:
    """Return published population evidence, never a breed 'safe limit'."""

    index, note = _catalog()
    row = index.get(_normalize(breed))
    if row is None:
        return None
    return BreedEvidence(
        name=str(row["name"]),
        odds_ratio_vs_labrador=float(row["odds_ratio"]),
        significantly_higher=bool(row["significantly_higher"]),
        evidence_note=note,
    )

