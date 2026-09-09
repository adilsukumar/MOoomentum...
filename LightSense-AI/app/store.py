import json
import sqlite3
from pathlib import Path
from threading import Lock

from app.models import (
    CustomLightsState,
    CustomLightsUpdate,
    HealthLightState,
    HealthStatusUpdate,
    RGB,
    health_color,
    utc_now,
)


class StateStore:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self._lock = Lock()

    def initialize(self) -> None:
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS light_state (
                    location TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        if self._read("custom") is None:
            self.set_custom(
                CustomLightsUpdate(pixels=[RGB(r=0, g=0, b=0)] * 3, brightness=255)
            )
        if self._read("status") is None:
            self.set_status(HealthStatusUpdate(health_score=0, is_charging=False))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, check_same_thread=False)

    def _read(self, location: str) -> dict | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM light_state WHERE location = ?", (location,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, location: str, payload: dict) -> None:
        encoded = json.dumps(payload)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO light_state(location, payload, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(location) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (location, encoded, payload["updated_at"]),
            )

    def get_custom(self) -> CustomLightsState:
        return CustomLightsState.model_validate(self._read("custom"))

    def set_custom(self, update: CustomLightsUpdate) -> CustomLightsState:
        state = CustomLightsState(**update.model_dump(), updated_at=utc_now())
        self._write("custom", state.model_dump(mode="json"))
        return state

    def get_status(self) -> HealthLightState:
        return HealthLightState.model_validate(self._read("status"))

    def set_status(self, update: HealthStatusUpdate) -> HealthLightState:
        color, reason = health_color(update.health_score, update.is_charging)
        state = HealthLightState(
            **update.model_dump(), color=color, reason=reason, updated_at=utc_now()
        )
        self._write("status", state.model_dump(mode="json"))
        return state

