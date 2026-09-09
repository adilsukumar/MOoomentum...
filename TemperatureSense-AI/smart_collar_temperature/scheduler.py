from __future__ import annotations

from dataclasses import dataclass

from .models import SensorSchedule, SleepState


@dataclass(frozen=True)
class SamplingPolicy:
    awake_environment_interval_s: int = 30
    sleeping_surface_interval_s: int = 60
    sleep_confirmation_s: int = 300
    # None exactly follows the requested policy: SHT40 is not sampled in sleep.
    # A low-rate value such as 300 is safer and still power efficient.
    sleeping_environment_interval_s: int | None = None


class TemperatureScheduler:
    """Select the temperature sensor from a BMI270 model-derived sleep state."""

    def __init__(self, policy: SamplingPolicy = SamplingPolicy()) -> None:
        self.policy = policy

    @staticmethod
    def _due(now_s: float, last_s: float | None, interval_s: int) -> bool:
        return last_s is None or now_s - last_s >= interval_s

    def schedule(
        self,
        *,
        now_s: float,
        bmi270_state: SleepState,
        state_since_s: float,
        last_sht40_s: float | None,
        last_ntc_s: float | None,
    ) -> SensorSchedule:
        state_age_s = max(0.0, now_s - state_since_s)
        sleep_confirmed = (
            bmi270_state is SleepState.SLEEPING
            and state_age_s >= self.policy.sleep_confirmation_s
        )

        if sleep_confirmed:
            read_ntc = self._due(
                now_s, last_ntc_s, self.policy.sleeping_surface_interval_s
            )
            sleep_env_interval = self.policy.sleeping_environment_interval_s
            read_sht40 = (
                sleep_env_interval is not None
                and self._due(now_s, last_sht40_s, sleep_env_interval)
            )
            return SensorSchedule(
                read_sht40=read_sht40,
                read_ntc=read_ntc,
                sleep_confirmed=True,
                reason="confirmed sleep: sample contact NTC; environment is power-managed",
            )

        # UNKNOWN and an unconfirmed sleep transition are handled as awake so
        # that a bad sleep classifier never enables surface-contact readings.
        read_sht40 = self._due(
            now_s, last_sht40_s, self.policy.awake_environment_interval_s
        )
        reason = (
            "awake: monitor environment"
            if bmi270_state is SleepState.AWAKE
            else "sleep unknown/unconfirmed: environment only"
        )
        return SensorSchedule(
            read_sht40=read_sht40,
            read_ntc=False,
            sleep_confirmed=False,
            reason=reason,
        )
