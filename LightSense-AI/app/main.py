from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.models import (
    AllLightsState,
    CustomLightsState,
    CustomLightsUpdate,
    HealthLightState,
    HealthStatusUpdate,
)
from app.mqtt import Publisher, build_publisher
from app.store import StateStore


def create_app(
    settings: Settings | None = None,
    store: StateStore | None = None,
    publisher: Publisher | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    state_store = store or StateStore(settings.database_path)
    mqtt_publisher = publisher or build_publisher(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state_store.initialize()
        mqtt_publisher.start()
        app.state.store = state_store
        app.state.publisher = mqtt_publisher
        yield
        mqtt_publisher.stop()

    app = FastAPI(
        title="LightSense WS2812B API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/lights", response_model=AllLightsState)
    def get_lights(request: Request) -> AllLightsState:
        return AllLightsState(
            custom=request.app.state.store.get_custom(),
            status=request.app.state.store.get_status(),
        )

    @app.put("/api/v1/lights/custom", response_model=CustomLightsState)
    def update_custom(update: CustomLightsUpdate, request: Request) -> CustomLightsState:
        state = request.app.state.store.set_custom(update)
        request.app.state.publisher.publish_custom(state)
        return state

    @app.put("/api/v1/lights/status", response_model=HealthLightState)
    def update_status(update: HealthStatusUpdate, request: Request) -> HealthLightState:
        state = request.app.state.store.set_status(update)
        request.app.state.publisher.publish_status(state)
        return state

    return app


app = create_app()

