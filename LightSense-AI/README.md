# LightSense WS2812B backend

This repository controls two physically separate WS2812B installations:

- `custom`: exactly three addressable pixels, each independently configurable across the RGB spectrum.
- `status`: one pixel derived by the backend from collar state. Charging is blue; otherwise health above 65 is green and health at or below 65 is red.

The backend persists desired state in SQLite and publishes retained MQTT commands. Retained messages make an ESP32 restore the latest desired state after it reconnects.

## Architecture

```text
Future web UI / collar health service
                 |
                 | HTTP JSON
                 v
          FastAPI backend ---- SQLite
                 |
                 | retained MQTT
          +------+------+
          |             |
   ESP32 custom    ESP32 status
    3 pixels         1 pixel
```

The two locations require a network-connected controller each. The included firmware targets ESP32 boards. A WS2812B by itself cannot receive HTTP or MQTT.

## Run the backend

Python 3.11+ and Docker are expected.

```powershell
Copy-Item .env.example .env
docker compose up -d mqtt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Change `.env` to enable MQTT:

```dotenv
LIGHTSENSE_MQTT_ENABLED=true
LIGHTSENSE_MQTT_HOST=localhost
```

Then start the service:

```powershell
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API. CORS is already enabled for common local front-end ports and can be changed with `LIGHTSENSE_CORS_ORIGINS`.

## API examples

Set the three custom pixels independently:

```powershell
$body = @{
  pixels = @(
    @{ r = 255; g = 0; b = 120 },
    @{ r = 0; g = 255; b = 80 },
    @{ r = 20; g = 40; b = 255 }
  )
  brightness = 180
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Put -Uri http://localhost:8000/api/v1/lights/custom -ContentType application/json -Body $body
```

Update the status pixel from collar data:

```powershell
$body = @{ health_score = 82; is_charging = $false; brightness = 180 } | ConvertTo-Json
Invoke-RestMethod -Method Put -Uri http://localhost:8000/api/v1/lights/status -ContentType application/json -Body $body
```

Read both desired states:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/lights
```

## Flash each ESP32

Install PlatformIO, then edit `firmware/platformio.ini` with the Wi-Fi credentials and the LAN IP of the computer running MQTT. Flash one controller per location:

```powershell
platformio run -d firmware -e custom_three -t upload
platformio run -d firmware -e status_one -t upload
```

For production, do not commit credentials in `platformio.ini`; use a private PlatformIO override or build-time secrets.

## Wiring

For each location:

- ESP32 GPIO 5 -> 330-500 ohm resistor -> WS2812B `DIN`.
- External regulated 5 V -> WS2812B `5V`.
- Power-supply ground -> WS2812B `GND` and ESP32 `GND` (common ground is required).
- Put a 500-1000 uF capacitor across 5 V and ground near the pixels.
- For reliable operation, use a 3.3 V-to-5 V logic-level shifter such as 74AHCT125 on the data line.

Do not power a multi-pixel installation from an ESP32 pin. Budget up to 60 mA per pixel at full white (180 mA for the three-pixel location), plus controller overhead.

## Tests

```powershell
pytest
```

