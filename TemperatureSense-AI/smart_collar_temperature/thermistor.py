from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class DividerOrientation(str, Enum):
    """Physical order of the voltage divider from Vref to ground."""

    FIXED_TO_VREF_NTC_TO_GROUND = "fixed_to_vref_ntc_to_ground"
    NTC_TO_VREF_FIXED_TO_GROUND = "ntc_to_vref_fixed_to_ground"


@dataclass(frozen=True)
class ThermistorConfig:
    nominal_resistance_ohm: float = 10_000.0
    nominal_temperature_c: float = 25.0
    beta_k: float = 3950.0
    fixed_resistor_ohm: float = 10_000.0
    adc_max: int = 4095
    orientation: DividerOrientation = DividerOrientation.FIXED_TO_VREF_NTC_TO_GROUND
    calibration_offset_c: float = 0.0


def adc_to_resistance(adc_count: int, config: ThermistorConfig = ThermistorConfig()) -> float:
    """Convert a ratiometric ADC count to NTC resistance.

    The result does not depend on Vref when the ADC and divider share Vref.
    End-point values are rejected because they represent an open/short circuit
    or a saturated ADC, not a usable temperature.
    """

    if not 0 < adc_count < config.adc_max:
        raise ValueError("ADC count is saturated; check for an open/short circuit")
    ratio = adc_count / config.adc_max
    if config.orientation is DividerOrientation.FIXED_TO_VREF_NTC_TO_GROUND:
        return config.fixed_resistor_ohm * ratio / (1.0 - ratio)
    return config.fixed_resistor_ohm * (1.0 - ratio) / ratio


def resistance_to_celsius(
    resistance_ohm: float, config: ThermistorConfig = ThermistorConfig()
) -> float:
    """Convert resistance with the single-beta equation.

    Use the actual beta (or preferably Steinhart-Hart coefficients) from the
    exact thermistor datasheet.  The default beta=3950 K is only a common 10 kΩ
    example and is not universal.
    """

    if resistance_ohm <= 0:
        raise ValueError("Thermistor resistance must be positive")
    t0_kelvin = config.nominal_temperature_c + 273.15
    inverse_t = (1.0 / t0_kelvin) + (
        math.log(resistance_ohm / config.nominal_resistance_ohm) / config.beta_k
    )
    return (1.0 / inverse_t) - 273.15 + config.calibration_offset_c


def adc_to_celsius(adc_count: int, config: ThermistorConfig = ThermistorConfig()) -> float:
    return resistance_to_celsius(adc_to_resistance(adc_count, config), config)

