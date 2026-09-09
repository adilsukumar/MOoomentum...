#include <assert.h>
#include <stdint.h>

#include "SmartCollarEnvironment.h"
#include "SmartCollarSurfaceTemperature.h"
#include "SmartCollarTemperatureGate.h"
#include "SmartCollarTemperatureLabels.h"

using smart_collar::SmartCollarTemperatureGate;
using smart_collar::TemperatureGateDecision;
using smart_collar::DividerOrientation;
using smart_collar::DogThermalProfile;
using smart_collar::EnvironmentAssessment;
using smart_collar::EnvironmentContext;
using smart_collar::EnvironmentRange;
using smart_collar::EnvironmentSeverity;
using smart_collar::SmartCollarSurfaceTemperature;
using smart_collar::SmartCollarEnvironment;
using smart_collar::SurfaceRange;
using smart_collar::SurfaceSeverity;
using smart_collar::SurfaceThresholdConfig;
using smart_collar::ThermistorSurfaceConfig;

int main() {
  SmartCollarTemperatureGate gate;

  TemperatureGateDecision decision = gate.update(false, 0UL);
  assert(decision.readSHT40);
  assert(!decision.readNTC);
  gate.markSHT40Read(0UL);

  decision = gate.update(true, 60000UL);
  assert(decision.sleepCandidate);
  assert(!decision.sleepConfirmed);
  assert(!decision.readNTC);
  gate.markSHT40Read(60000UL);

  decision = gate.update(true, 359999UL);
  assert(!decision.sleepConfirmed);
  assert(!decision.readNTC);

  decision = gate.update(true, 360000UL);
  assert(decision.sleepConfirmed);
  assert(decision.readNTC);
  assert(!decision.readSHT40);
  gate.markNTCRead(360000UL);

  decision = gate.update(true, 419999UL);
  assert(!decision.readNTC);
  decision = gate.update(true, 420000UL);
  assert(decision.readNTC);

  decision = gate.update(false, 420001UL);
  assert(!decision.readNTC);
  assert(decision.readSHT40);

  // Confirm scheduling remains correct across uint32_t millisecond rollover.
  gate.reset();
  const uint32_t beforeRollover = UINT32_MAX - 10000UL;
  decision = gate.update(false, beforeRollover);
  assert(decision.readSHT40);
  gate.markSHT40Read(beforeRollover);
  decision = gate.update(false, 25000UL);
  assert(decision.readSHT40);

  const ThermistorSurfaceConfig thermistorConfig(
      10000.0f, 25.0f, 3950.0f, 10000.0f, 4095,
      DividerOrientation::FixedToVrefNtcToGround);
  const SmartCollarSurfaceTemperature surfaceTemperature(thermistorConfig);

  float convertedSurfaceC = 0.0f;
  assert(surfaceTemperature.adcToSurfaceC(2048, convertedSurfaceC));
  assert(convertedSurfaceC > 24.9f && convertedSurfaceC < 25.1f);
  assert(!surfaceTemperature.adcToSurfaceC(0, convertedSurfaceC));
  assert(!surfaceTemperature.adcToSurfaceC(4095, convertedSurfaceC));
  assert(surfaceTemperature.voltageToSurfaceC(1650.0f, 3300.0f,
                                               convertedSurfaceC));
  assert(convertedSurfaceC > 24.9f && convertedSurfaceC < 25.1f);

  const float stableAboveSamples[] = {34.0f, 34.1f, 34.0f,
                                      34.1f, 34.0f};
  smart_collar::SurfaceAssessment surfaceAssessment =
      surfaceTemperature.assess(stableAboveSamples, 5, true, 33.0f);
  assert(surfaceAssessment.range == SurfaceRange::AboveBaseline);
  assert(surfaceAssessment.severity == SurfaceSeverity::Caution);
  assert(surfaceAssessment.contactStable);

  const float unstableSamples[] = {32.0f, 32.5f, 33.0f, 33.5f, 34.0f};
  surfaceAssessment =
      surfaceTemperature.assess(unstableSamples, 5, true, 33.0f);
  assert(surfaceAssessment.range == SurfaceRange::ContactUnstable);
  assert(surfaceAssessment.severity == SurfaceSeverity::Invalid);

  surfaceAssessment =
      surfaceTemperature.assess(stableAboveSamples, 5, false, 0.0f);
  assert(surfaceAssessment.range == SurfaceRange::BaselineRequired);

  const float healthyNightMedians[] = {33.0f, 33.2f, 32.9f, 33.1f,
                                       33.0f, 33.3f, 33.1f};
  float learnedBaselineC = 0.0f;
  assert(surfaceTemperature.calculateHealthyBaseline(
      healthyNightMedians, 7, learnedBaselineC));
  assert(learnedBaselineC > 33.09f && learnedBaselineC < 33.11f);

  SurfaceThresholdConfig customSurfaceThresholds;
  customSurfaceThresholds.normalDeltaC = 0.5f;
  customSurfaceThresholds.highDeltaC = 0.9f;
  const SmartCollarSurfaceTemperature sensitiveSurfaceTemperature(
      thermistorConfig, customSurfaceThresholds);
  surfaceAssessment =
      sensitiveSurfaceTemperature.assess(stableAboveSamples, 5, true, 33.0f);
  assert(surfaceAssessment.range == SurfaceRange::WellAboveBaseline);
  assert(surfaceAssessment.severity == SurfaceSeverity::High);

  SmartCollarEnvironment environment;
  DogThermalProfile dogProfile;
  EnvironmentContext environmentContext;

  EnvironmentAssessment environmentAssessment =
      environment.update(22.0f, 45.0f, 0UL, dogProfile, environmentContext);
  assert(environmentAssessment.range == EnvironmentRange::Normal);
  assert(environmentAssessment.severity == EnvironmentSeverity::Normal);
  assert(smart_collar::environmentRangeLabel(environmentAssessment.range)[0] ==
         'N');
  assert(smart_collar::environmentActionLabel(
             environmentAssessment.severity)[0] == 'C');
  assert(!environmentAssessment.rapidRiseDetected);

  environmentAssessment =
      environment.update(26.0f, 45.0f, 30000UL, dogProfile,
                         environmentContext);
  assert(environmentAssessment.rapidRiseDetected);
  assert(environmentAssessment.immediateThresholdBypass);
  assert(environmentAssessment.range == EnvironmentRange::Warm);
  assert(environmentAssessment.severity == EnvironmentSeverity::High);

  // An extreme heat reading bypasses filtering immediately; this models the
  // intended visible spike when a strong nearby heat source reaches the SHT40.
  environmentAssessment =
      environment.update(36.0f, 45.0f, 60000UL, dogProfile,
                         environmentContext);
  assert(environmentAssessment.range == EnvironmentRange::ExtremeHigh);
  assert(environmentAssessment.severity == EnvironmentSeverity::Critical);
  assert(environmentAssessment.immediateThresholdBypass);

  environment.reset();
  environmentAssessment =
      environment.update(-6.0f, 45.0f, 0UL, dogProfile, environmentContext);
  assert(environmentAssessment.range == EnvironmentRange::ExtremeLow);
  assert(environmentAssessment.severity == EnvironmentSeverity::Critical);

  environment.reset();
  dogProfile.brachycephalic = true;
  environmentContext.activity = smart_collar::CollarActivity::Active;
  environmentAssessment =
      environment.update(26.0f, 85.0f, 0UL, dogProfile, environmentContext);
  assert(environmentAssessment.range == EnvironmentRange::Warm);
  assert(environmentAssessment.severity == EnvironmentSeverity::Critical);

  environmentAssessment =
      environment.update(200.0f, 45.0f, 1000UL, dogProfile,
                         environmentContext);
  assert(environmentAssessment.range == EnvironmentRange::SensorInvalid);
  assert(environmentAssessment.severity == EnvironmentSeverity::Invalid);
  assert(smart_collar::surfaceRangeLabel(SurfaceRange::AboveBaseline)[0] ==
         'A');
  assert(smart_collar::surfaceActionLabel(SurfaceSeverity::High)[0] == 'C');

  return 0;
}
