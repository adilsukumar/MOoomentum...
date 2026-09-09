from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class Severity(IntEnum):
    NORMAL = 0
    CAUTION = 1
    HIGH = 2
    CRITICAL = 3
    INVALID = 4


class SleepState(str, Enum):
    AWAKE = "awake"
    SLEEPING = "sleeping"
    UNKNOWN = "unknown"


class ActivityLevel(str, Enum):
    RESTING = "resting"
    LIGHT = "light"
    ACTIVE = "active"
    INTENSE = "intense"


class SkullShape(str, Enum):
    BRACHYCEPHALIC = "brachycephalic"
    MESOCEPHALIC = "mesocephalic"
    DOLICHOCEPHALIC = "dolichocephalic"
    UNKNOWN = "unknown"


class CoatType(str, Enum):
    HAIRLESS = "hairless"
    SHORT_SINGLE = "short_single"
    LONG_SINGLE = "long_single"
    DOUBLE = "double"
    HEAVY_DOUBLE = "heavy_double"
    UNKNOWN = "unknown"


class SizeClass(str, Enum):
    TOY = "toy"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    GIANT = "giant"
    UNKNOWN = "unknown"


class BodyCondition(str, Enum):
    UNDERWEIGHT = "underweight"
    IDEAL = "ideal"
    OVERWEIGHT = "overweight"
    OBESE = "obese"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DogProfile:
    """Risk traits work for purebred and mixed-breed dogs.

    `healthy_sleep_surface_baseline_c` is the median collar-contact surface
    temperature collected for this exact dog, fit, and sensor site over healthy
    nights. It is never converted to internal/core temperature.
    """

    breed: str = "unknown"
    age_years: float | None = None
    size: SizeClass = SizeClass.UNKNOWN
    skull: SkullShape = SkullShape.UNKNOWN
    coat: CoatType = CoatType.UNKNOWN
    body_condition: BodyCondition = BodyCondition.UNKNOWN
    respiratory_or_cardiac_disease: bool = False
    heat_acclimated: bool = False
    cold_acclimated: bool = False
    healthy_sleep_surface_baseline_c: float | None = None


@dataclass(frozen=True)
class EnvironmentContext:
    activity: ActivityLevel = ActivityLevel.RESTING
    direct_sun: bool = False
    poor_airflow: bool = False
    enclosed_vehicle: bool = False


@dataclass(frozen=True)
class Alert:
    severity: Severity
    code: str
    range_label: str
    message: str
    actions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorSchedule:
    read_sht40: bool
    read_ntc: bool
    sleep_confirmed: bool
    reason: str


@dataclass(frozen=True)
class RabiesContext:
    known_or_possible_exposure: bool = False
    abrupt_behavior_change: bool = False
    neurologic_signs: bool = False
    swallowing_difficulty_or_excess_salivation: bool = False
    aggression_or_unusual_tameness: bool = False
