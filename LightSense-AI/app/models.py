from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

RGBChannel = Annotated[int, Field(ge=0, le=255)]


class RGB(BaseModel):
    r: RGBChannel
    g: RGBChannel
    b: RGBChannel

    @classmethod
    def from_hex(cls, value: str) -> "RGB":
        normalized = value.removeprefix("#")
        if len(normalized) != 6:
            raise ValueError("Color must be a six-digit hex value")
        try:
            return cls(
                r=int(normalized[0:2], 16),
                g=int(normalized[2:4], 16),
                b=int(normalized[4:6], 16),
            )
        except ValueError as exc:
            raise ValueError("Color must be a valid hex value") from exc

    def as_hex(self) -> str:
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"


class CustomLightsUpdate(BaseModel):
    pixels: list[RGB] = Field(min_length=3, max_length=3)
    brightness: int = Field(default=255, ge=0, le=255)

    @field_validator("pixels")
    @classmethod
    def require_three_pixels(cls, pixels: list[RGB]) -> list[RGB]:
        if len(pixels) != 3:
            raise ValueError("Exactly three pixel colors are required")
        return pixels


class HealthStatusUpdate(BaseModel):
    health_score: float = Field(ge=0, le=100)
    is_charging: bool
    brightness: int = Field(default=255, ge=0, le=255)


class CustomLightsState(CustomLightsUpdate):
    location: Literal["custom"] = "custom"
    updated_at: datetime


class HealthLightState(HealthStatusUpdate):
    location: Literal["status"] = "status"
    color: RGB
    reason: Literal["charging", "healthy", "low_health"]
    updated_at: datetime


class AllLightsState(BaseModel):
    custom: CustomLightsState
    status: HealthLightState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def health_color(health_score: float, is_charging: bool) -> tuple[RGB, str]:
    """Charging wins; otherwise only a score strictly above 65 is green."""
    if is_charging:
        return RGB(r=0, g=0, b=255), "charging"
    if health_score > 65:
        return RGB(r=0, g=255, b=0), "healthy"
    return RGB(r=255, g=0, b=0), "low_health"

