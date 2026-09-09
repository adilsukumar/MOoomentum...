"""Smart-collar temperature reference engine.

This package emits risk bands and actions.  It is not a diagnostic device and
must not be used to claim that a dog has fever, heatstroke, or rabies.
"""

from .breeds import BreedEvidence, get_breed_evidence
from .models import (
    ActivityLevel,
    Alert,
    BodyCondition,
    CoatType,
    DogProfile,
    EnvironmentContext,
    RabiesContext,
    SensorSchedule,
    Severity,
    SkullShape,
    SleepState,
    SizeClass,
)
from .safety import (
    assess_environment,
    assess_rabies_context,
    assess_surface_contact_temperature,
)
from .scheduler import SamplingPolicy, TemperatureScheduler
from .thermistor import DividerOrientation, ThermistorConfig, adc_to_celsius

__all__ = [
    "ActivityLevel",
    "Alert",
    "BodyCondition",
    "BreedEvidence",
    "CoatType",
    "DividerOrientation",
    "DogProfile",
    "EnvironmentContext",
    "RabiesContext",
    "SamplingPolicy",
    "SensorSchedule",
    "Severity",
    "SizeClass",
    "SkullShape",
    "SleepState",
    "TemperatureScheduler",
    "ThermistorConfig",
    "adc_to_celsius",
    "assess_environment",
    "assess_rabies_context",
    "assess_surface_contact_temperature",
    "get_breed_evidence",
]
