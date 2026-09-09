# ESP32-C3 SuperMini / ESP-IDF integration contract

The trained MotionSense code owns all BMI270 acquisition, preprocessing,
inference, labels, and confidence handling. The temperature gate sees exactly
one value from that system:

```cpp
bool dogIsSleeping;
```

No class number or model label is hard-coded in the temperature library.

## Add the ESP-IDF component

Copy `esp32c3/components/smart_collar_temperature` into the firmware project's
`components` directory. ESP-IDF discovers its `CMakeLists.txt` automatically.

At firmware scope:

```cpp
#include <SmartCollarTemperatureGate.h>

smart_collar::SmartCollarTemperatureGate temperatureGate;
```

At the point where the existing AI code has made its final classification,
retain that result as a boolean named `dogIsSleeping`. Use ESP-IDF's monotonic
timer when calling the gate:

```cpp
#include "esp_timer.h"

const uint32_t nowMs =
    static_cast<uint32_t>(esp_timer_get_time() / 1000ULL);
const smart_collar::TemperatureGateDecision temperatureDecision =
    temperatureGate.update(dogIsSleeping, nowMs);

if (temperatureDecision.readSHT40) {
  if (readAndStoreSHT40()) {
    temperatureGate.markSHT40Read(nowMs);
  }
}

if (temperatureDecision.readNTC) {
  if (readAndStoreNTC()) {
    temperatureGate.markNTCRead(nowMs);
  }
}
```

`readAndStoreSHT40()` and `readAndStoreNTC()` in that block represent the
hardware team's existing sensor functions and should return `true` only when a
reading passes the driver's error checks. The gate does not own I²C, ADC pins,
or a particular sensor library.

## Resulting behavior

| Final MotionSense boolean | Gate behavior |
|---|---|
| `false` | NTC off; SHT40 requested every 30 seconds |
| first change to `true` | start sleep-confirmation timer; NTC remains off |
| continuously `true` for 5 minutes | NTC requested every 60 seconds; SHT40 off |
| changes back to `false` | NTC disabled immediately; SHT40 resumes |

Any inference that the AI code considers invalid, uncertain, stale, or below its
confidence requirement must be converted to `false`. That is the fail-safe
choice because it prevents an unconfirmed surface-contact measurement.

The model result should be refreshed often enough that a wake transition is not
delayed. The gate uses rollover-safe `uint32_t` millisecond arithmetic.

## Surface-contact calculation

`SmartCollarSurfaceTemperature.h` performs the NTC calculations without any
internal-temperature conversion. Construct it with the exact values from the
chosen thermistor, fixed resistor, ADC, and divider wiring:

```cpp
#include <SmartCollarSurfaceTemperature.h>

const smart_collar::ThermistorSurfaceConfig thermistorConfig(
    nominalResistanceOhm,
    nominalTemperatureC,
    betaK,
    fixedResistorOhm,
    adcMaximum,
    dividerOrientation,
    measuredCalibrationOffsetC);

smart_collar::SmartCollarSurfaceTemperature surfaceTemperature(
    thermistorConfig);
```

The hardware values in this constructor must come from the actual schematic,
ADC configuration, thermistor datasheet, and bench calibration; the library
does not guess them.

On the ESP32-C3, use ESP-IDF's ADC oneshot driver and calibration driver. The
preferred conversion accepts calibrated divider-node voltage plus the measured
divider supply voltage:

```cpp
float surfaceC;
if (surfaceTemperature.voltageToSurfaceC(
        calibratedDividerMillivolts,
        measuredDividerSupplyMillivolts,
        surfaceC)) {
  // Add surfaceC to the current confirmed-sleep sample window.
}
```

`adcToSurfaceC()` remains available when the ADC/divider has been characterized
as a ratiometric raw-count system. ESP32 ADC nonlinearity makes the calibrated
voltage path preferable.

After collecting at least five confirmed-sleep samples, pass them with the
dog's stored healthy surface baseline:

```cpp
const smart_collar::SurfaceAssessment assessment =
    surfaceTemperature.assess(
        surfaceSamples,
        surfaceSampleCount,
        surfaceBaselineAvailable,
        healthySleepSurfaceBaselineC);
```

The result contains surface median, sample span/contact stability, difference
from personal baseline, range, and severity. Exact values may be logged for
validation while the user interface shows only the range and action. The result
does not contain internal temperature, fever grade, or a disease probability.

### Personal surface baseline and bands

There is no universal normal neck-surface temperature for every dog. Establish
the baseline from one stable-window median on each of at least seven known-
healthy sleeping nights:

```cpp
float healthySleepSurfaceBaselineC;
const bool baselineReady = surfaceTemperature.calculateHealthyBaseline(
    healthyNightWindowMedians,
    healthyNightCount,
    healthySleepSurfaceBaselineC);
```

Persist the resulting baseline with the dog and collar-fit/site profile. Relearn
it after the NTC position or collar fit materially changes. With baseline `B`,
the default surface bands are:

| Filtered stable surface result | Range | Severity |
|---|---|---|
| below `B - 1.5 °C` | well below baseline | high |
| `B - 1.5` to below `B - 0.8 °C` | below baseline | caution |
| `B - 0.8` through `B + 0.8 °C` | personal baseline | normal |
| above `B + 0.8` through `B + 1.5 °C` | above baseline | caution |
| above `B + 1.5 °C` | well above baseline | high |

Five to sixteen samples are median-filtered. A sample span greater than 1.0 °C
is reported as `CONTACT_UNSTABLE`, not mistaken for a health change. All three
deltas can be changed through `SurfaceThresholdConfig` after field validation.

## SHT40 environmental calculation

Pass each successful SHT40 air-temperature and relative-humidity result to the
environment engine:

```cpp
#include <SmartCollarEnvironment.h>
#include <SmartCollarTemperatureLabels.h>

smart_collar::SmartCollarEnvironment environment;
smart_collar::DogThermalProfile dogThermalProfile;
smart_collar::EnvironmentContext environmentContext;

const smart_collar::EnvironmentAssessment environmentResult =
    environment.update(
        shtTemperatureC,
        shtHumidityPct,
        static_cast<uint32_t>(esp_timer_get_time() / 1000ULL),
        dogThermalProfile,
        environmentContext);

ESP_LOGI("temperature", "range=%s severity=%s action=%s",
         smart_collar::environmentRangeLabel(environmentResult.range),
         smart_collar::environmentSeverityLabel(environmentResult.severity),
         smart_collar::environmentActionLabel(environmentResult.severity));
```

Set the dog profile once from onboarding data. Set activity from the existing
MotionSense result, and set sun/enclosure context when the product has a source
for those facts. Unknown facts remain `false`; the library does not invent them.
The profile supports skull shape, coat, size, weight condition, vulnerable age,
respiratory/cardiac conditions, acclimation, and optional additional risk points
from the existing breed-evidence catalog. Keep additional points small and
veterinarian-reviewed; they are severity modifiers, not degrees Celsius.

Default SHT40 air bands:

| SHT40 air value | Range | Base severity |
|---:|---|---|
| below -5 °C | extreme low | critical |
| -5 to below 0 °C | very low | high |
| 0 to below 10 °C | low | caution |
| 10 to below 25 °C | normal | normal |
| 25 to below 30 °C | warm | caution |
| 30 to below 35 °C | high | high |
| 35 °C and above | extreme high | critical |

Humidity, MotionSense activity, sunlight, poor airflow, vehicle confinement,
and dog vulnerability can raise the final severity. A rise or fall of at least
2 °C/minute bypasses smoothing and raises severity. Any raw reading at or beyond
an extreme boundary also bypasses smoothing immediately.

## Controlled high-heat demonstration

For a bench demonstration only, reduce SHT40 sampling to one second so a short
heat pulse is not missed:

```cpp
smart_collar::TemperatureGateConfig demonstrationConfig;
demonstrationConfig.awakeEnvironmentIntervalMs = 1000UL;
smart_collar::SmartCollarTemperatureGate temperatureGate(demonstrationConfig);
```

To exercise **both** sensors during a hardware-only bench test, use this test
configuration and feed `true` to `update()`:

```cpp
smart_collar::TemperatureGateConfig benchConfig;
benchConfig.sleepConfirmationMs = 0UL;
benchConfig.sleepingSurfaceIntervalMs = 1000UL;
benchConfig.sleepingEnvironmentIntervalMs = 1000UL;
smart_collar::SmartCollarTemperatureGate temperatureGate(benchConfig);

const smart_collar::TemperatureGateDecision testDecision =
    temperatureGate.update(
        true, static_cast<uint32_t>(esp_timer_get_time() / 1000ULL));
```

This intentionally overrides the production duty cycle. Restore the default
gate for animal use. The SHT40 responds to its local air/package heating; the
NTC is a contact sensor and should be tested against a controlled warm surface,
not expected to represent nearby air heat accurately.

Start near room temperature, then bring a controlled radiant/hot-air source
nearer without touching the sensor. The ESP-IDF log should progress through
`NORMAL`, `WARM`, `HIGH`, and `EXTREME_HIGH` if those air readings are actually
reached. A sudden 2 °C/minute rise sets `rapidRiseDetected`, and 35 °C or above
is immediately `CRITICAL` without waiting for the filter.

Do not apply an open flame to the SHT40, NTC, collar, wiring, or battery. A flame
tests fire damage and radiant heating—not ambient-temperature accuracy. Run the
demonstration with no animal or battery present, on a fire-resistant bench, and
use an independent reference thermometer. Restore the deployed sampling
interval after the test.

## Optional sleeping environment heartbeat

The requested default switches SHT40 measurements off during sleep. To retain a
low-power five-minute environmental safety check, construct the gate as follows:

```cpp
smart_collar::TemperatureGateConfig gateConfig;
gateConfig.sleepingEnvironmentIntervalMs = 300000UL;
smart_collar::SmartCollarTemperatureGate temperatureGate(gateConfig);
```
