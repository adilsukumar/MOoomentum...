# Smart Collar Temperature Safety Engine

This repository is a source-backed reference implementation for two sensors:

- **SHT40**: local air temperature and relative humidity while the dog is awake.
- **10 kΩ NTC thermistor**: collar-to-skin contact-temperature trends only after
  the BMI270 has classified stable sleep.

The engine returns **ranges and actions**, not a displayed exact temperature.
It deliberately does not diagnose fever, heatstroke, or rabies.

## The important design correction

The thermistor is designed strictly as a **surface-contact sensor**. Fur, skin
perfusion, room temperature, collar pressure, contact site, and movement all
change its reading. The engine compares stable sleeping measurements with that
individual dog's healthy surface-contact baseline. It contains no conversion
to internal/core temperature and no absolute fever thresholds.

Likewise, there is no validated table of safe upper and lower air temperatures
for every breed. Published evidence provides population risk factors, not breed
cut-offs. Mixed breeds also cannot be handled reliably by a breed-name lookup.
The engine combines local temperature, humidity, activity, sun/airflow context,
and dog traits (skull, coat, size, age, body condition, and disease).

Read [the threshold and validation guide](docs/thresholds-and-validation.md)
before using the code on an animal.

## Quick use

```python
from smart_collar_temperature import (
    ActivityLevel,
    CoatType,
    DogProfile,
    EnvironmentContext,
    SizeClass,
    assess_environment,
)

dog = DogProfile(
    breed="Siberian Husky",
    coat=CoatType.HEAVY_DOUBLE,
    size=SizeClass.LARGE,
)

alert = assess_environment(
    temperature_c=31.2,
    relative_humidity_pct=72.0,
    profile=dog,
    context=EnvironmentContext(activity=ActivityLevel.ACTIVE),
)

print(alert.range_label)  # e.g. "hot range"
print(alert.severity.name)
print(alert.actions)
```

The full scheduling and NTC example is in [examples/demo.py](examples/demo.py).

## Connecting your trained MotionSense model to ESP32-C3

The temperature module deliberately knows nothing about model labels, class
IDs, tensors, or confidence values. Your existing inference code reduces its
result to one final boolean: `dogIsSleeping`. Pass that boolean to the portable
ESP-IDF C++ gate, which returns `readSHT40` and `readNTC` decisions.

See the exact firmware contract in
[esp32c3/INTEGRATION.md](esp32c3/INTEGRATION.md). The ESP-IDF component includes
[sensor scheduling](esp32c3/components/smart_collar_temperature/include/SmartCollarTemperatureGate.h),
[SHT40 environmental assessment](esp32c3/components/smart_collar_temperature/include/SmartCollarEnvironment.h),
[NTC surface assessment](esp32c3/components/smart_collar_temperature/include/SmartCollarSurfaceTemperature.h),
and [user-facing labels](esp32c3/components/smart_collar_temperature/include/SmartCollarTemperatureLabels.h).

## BMI270 sensor policy

Default behavior exactly follows the requested power policy:

| BMI270 state | SHT40 | NTC |
|---|---:|---:|
| Awake | every 30 s | off |
| Unknown or sleep not stable for 5 min | every 30 s | off |
| Confirmed sleeping | off | every 60 s |

Turning the environment sensor fully off during sleep can miss a dangerous
sleeping environment. Set `sleeping_environment_interval_s=300` for a safer
low-rate check. The SHT40 itself is very low power; MCU wake time and radio use
may dominate the energy cost.

## Run the checks

```powershell
python -m unittest discover -s tests -v
python -m examples.demo
```

## Production status

This is a **prototype decision-support engine**. Before animal trials, get the
thresholds, wording, hardware contact design, and escalation workflow reviewed
by a veterinarian and the applicable medical-device/product-safety specialist.
