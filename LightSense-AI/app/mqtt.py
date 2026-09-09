import json
import logging
from typing import Protocol

import paho.mqtt.client as mqtt

from app.config import Settings
from app.models import CustomLightsState, HealthLightState

logger = logging.getLogger(__name__)


class Publisher(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def publish_custom(self, state: CustomLightsState) -> None: ...
    def publish_status(self, state: HealthLightState) -> None: ...


class NullPublisher:
    def start(self) -> None:
        logger.info("MQTT disabled; state changes will only be persisted")

    def stop(self) -> None:
        pass

    def publish_custom(self, state: CustomLightsState) -> None:
        pass

    def publish_status(self, state: HealthLightState) -> None:
        pass


class MqttPublisher:
    def __init__(self, settings: Settings):
        self.prefix = settings.mqtt_topic_prefix.strip("/")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if settings.mqtt_username:
            self.client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
        self.host = settings.mqtt_host
        self.port = settings.mqtt_port

    def start(self) -> None:
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def _publish(self, location: str, payload: dict) -> None:
        info = self.client.publish(
            f"{self.prefix}/{location}/set",
            json.dumps(payload, separators=(",", ":")),
            qos=1,
            retain=True,
        )
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed with code {info.rc}")

    def publish_custom(self, state: CustomLightsState) -> None:
        self._publish(
            "custom",
            {
                "pixels": [pixel.model_dump() for pixel in state.pixels],
                "brightness": state.brightness,
            },
        )

    def publish_status(self, state: HealthLightState) -> None:
        self._publish(
            "status",
            {"pixels": [state.color.model_dump()], "brightness": state.brightness},
        )


def build_publisher(settings: Settings) -> Publisher:
    return MqttPublisher(settings) if settings.mqtt_enabled else NullPublisher()

