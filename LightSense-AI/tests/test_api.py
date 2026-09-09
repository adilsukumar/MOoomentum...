from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import CustomLightsState, HealthLightState
from app.store import StateStore


class RecordingPublisher:
    def __init__(self):
        self.custom: CustomLightsState | None = None
        self.status: HealthLightState | None = None

    def start(self):
        pass

    def stop(self):
        pass

    def publish_custom(self, state):
        self.custom = state

    def publish_status(self, state):
        self.status = state


def make_client(tmp_path):
    publisher = RecordingPublisher()
    store = StateStore(str(tmp_path / "test.db"))
    app = create_app(
        settings=Settings(database_path=str(tmp_path / "test.db")),
        store=store,
        publisher=publisher,
    )
    return TestClient(app), publisher


def test_custom_location_controls_all_three_pixels(tmp_path):
    client, publisher = make_client(tmp_path)
    with client:
        response = client.put(
            "/api/v1/lights/custom",
            json={
                "pixels": [
                    {"r": 255, "g": 0, "b": 1},
                    {"r": 2, "g": 128, "b": 3},
                    {"r": 4, "g": 5, "b": 255},
                ],
                "brightness": 100,
            },
        )
    assert response.status_code == 200
    assert response.json()["pixels"][1] == {"r": 2, "g": 128, "b": 3}
    assert publisher.custom is not None
    assert publisher.custom.brightness == 100


def test_custom_location_requires_exactly_three_pixels(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        response = client.put(
            "/api/v1/lights/custom",
            json={"pixels": [{"r": 0, "g": 0, "b": 0}]},
        )
    assert response.status_code == 422


def test_health_above_65_is_green(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        response = client.put(
            "/api/v1/lights/status",
            json={"health_score": 65.01, "is_charging": False},
        )
    assert response.json()["color"] == {"r": 0, "g": 255, "b": 0}
    assert response.json()["reason"] == "healthy"


def test_health_at_65_is_red(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        response = client.put(
            "/api/v1/lights/status",
            json={"health_score": 65, "is_charging": False},
        )
    assert response.json()["color"] == {"r": 255, "g": 0, "b": 0}
    assert response.json()["reason"] == "low_health"


def test_charging_blue_overrides_health(tmp_path):
    client, publisher = make_client(tmp_path)
    with client:
        response = client.put(
            "/api/v1/lights/status",
            json={"health_score": 99, "is_charging": True, "brightness": 80},
        )
    assert response.json()["color"] == {"r": 0, "g": 0, "b": 255}
    assert response.json()["reason"] == "charging"
    assert publisher.status is not None
    assert publisher.status.color.b == 255

