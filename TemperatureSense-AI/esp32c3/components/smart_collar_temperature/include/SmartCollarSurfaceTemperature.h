#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace smart_collar {

enum class DividerOrientation : uint8_t {
  FixedToVrefNtcToGround,
  NtcToVrefFixedToGround
};

struct ThermistorSurfaceConfig {
  float nominalResistanceOhm;
  float nominalTemperatureC;
  float betaK;
  float fixedResistorOhm;
  uint16_t adcMaximum;
  DividerOrientation orientation;
  float calibrationOffsetC;

  ThermistorSurfaceConfig(float nominalResistance, float nominalTemperature,
                          float beta, float fixedResistance,
                          uint16_t maximumAdc,
                          DividerOrientation dividerOrientation,
                          float calibrationOffset = 0.0f)
      : nominalResistanceOhm(nominalResistance),
        nominalTemperatureC(nominalTemperature),
        betaK(beta),
        fixedResistorOhm(fixedResistance),
        adcMaximum(maximumAdc),
        orientation(dividerOrientation),
        calibrationOffsetC(calibrationOffset) {}
};

struct SurfaceThresholdConfig {
  float normalDeltaC;
  float highDeltaC;
  float maximumStableSpanC;

  SurfaceThresholdConfig()
      : normalDeltaC(0.8f), highDeltaC(1.5f), maximumStableSpanC(1.0f) {}
};

enum class SurfaceRange : uint8_t {
  BaselineRequired,
  TooFewSamples,
  SensorInvalid,
  ContactUnstable,
  WellBelowBaseline,
  BelowBaseline,
  PersonalBaseline,
  AboveBaseline,
  WellAboveBaseline
};

enum class SurfaceSeverity : uint8_t {
  Normal,
  Caution,
  High,
  Invalid
};

struct SurfaceAssessment {
  SurfaceRange range;
  SurfaceSeverity severity;
  float surfaceMedianC;
  float baselineDeltaC;
  float sampleSpanC;
  uint8_t sampleCount;
  bool contactStable;
};

class SmartCollarSurfaceTemperature {
 public:
  static const uint8_t kMinimumSamples = 5;
  static const uint8_t kMaximumSamples = 16;
  static const uint8_t kMinimumBaselineWindows = 7;
  static const uint8_t kMaximumBaselineWindows = 32;

  explicit SmartCollarSurfaceTemperature(
      const ThermistorSurfaceConfig& config,
      const SurfaceThresholdConfig& thresholds = SurfaceThresholdConfig())
      : config_(config), thresholds_(thresholds) {}

  bool adcToSurfaceC(uint16_t adcCount, float& surfaceC) const {
    if (!configIsValid() || adcCount == 0 || adcCount >= config_.adcMaximum) {
      return false;
    }

    const float ratio =
        static_cast<float>(adcCount) / static_cast<float>(config_.adcMaximum);
    return dividerRatioToSurfaceC(ratio, surfaceC);
  }

  // Prefer this ESP32-C3 path when ESP-IDF ADC calibration is enabled. Supply
  // the calibrated divider-node voltage and measured divider supply voltage.
  bool voltageToSurfaceC(float dividerMillivolts,
                         float dividerSupplyMillivolts,
                         float& surfaceC) const {
    if (!configIsValid() || !isfinite(dividerMillivolts) ||
        !isfinite(dividerSupplyMillivolts) ||
        dividerSupplyMillivolts <= 0.0f || dividerMillivolts <= 0.0f ||
        dividerMillivolts >= dividerSupplyMillivolts) {
      return false;
    }
    return dividerRatioToSurfaceC(
        dividerMillivolts / dividerSupplyMillivolts, surfaceC);
  }

 private:
  bool dividerRatioToSurfaceC(float ratio, float& surfaceC) const {
    if (!isfinite(ratio) || ratio <= 0.0f || ratio >= 1.0f) return false;
    float resistanceOhm = 0.0f;
    if (config_.orientation ==
        DividerOrientation::FixedToVrefNtcToGround) {
      resistanceOhm = config_.fixedResistorOhm * ratio / (1.0f - ratio);
    } else {
      resistanceOhm = config_.fixedResistorOhm * (1.0f - ratio) / ratio;
    }

    if (!isfinite(resistanceOhm) || resistanceOhm <= 0.0f) {
      return false;
    }

    const float nominalKelvin = config_.nominalTemperatureC + 273.15f;
    const float inverseTemperature =
        (1.0f / nominalKelvin) +
        (logf(resistanceOhm / config_.nominalResistanceOhm) / config_.betaK);
    if (!isfinite(inverseTemperature) || inverseTemperature <= 0.0f) {
      return false;
    }

    surfaceC = (1.0f / inverseTemperature) - 273.15f +
               config_.calibrationOffsetC;
    return isfinite(surfaceC) && surfaceC >= 0.0f && surfaceC <= 50.0f;
  }

 public:

  // Supply one stable-window median from each known-healthy sleeping night.
  // The robust median becomes this dog's personal surface baseline. Persist it
  // in flash/NVS together with the collar-fit/site version.
  bool calculateHealthyBaseline(const float* healthyWindowMediansC,
                                uint8_t windowCount,
                                float& baselineC) const {
    if (healthyWindowMediansC == nullptr ||
        windowCount < kMinimumBaselineWindows ||
        windowCount > kMaximumBaselineWindows) {
      return false;
    }
    float sorted[kMaximumBaselineWindows];
    for (uint8_t i = 0; i < windowCount; ++i) {
      const float value = healthyWindowMediansC[i];
      if (!isfinite(value) || value < 0.0f || value > 50.0f) return false;
      sorted[i] = value;
    }
    insertionSort(sorted, windowCount);
    baselineC = medianOfSorted(sorted, windowCount);
    return true;
  }

  SurfaceAssessment assess(const float* surfaceSamplesC, uint8_t sampleCount,
                           bool baselineAvailable,
                           float healthySleepSurfaceBaselineC) const {
    SurfaceAssessment result = {SurfaceRange::SensorInvalid,
                                SurfaceSeverity::Invalid,
                                0.0f,
                                0.0f,
                                0.0f,
                                sampleCount,
                                false};

    if (surfaceSamplesC == nullptr || sampleCount < kMinimumSamples ||
        sampleCount > kMaximumSamples) {
      result.range = SurfaceRange::TooFewSamples;
      return result;
    }

    float sorted[kMaximumSamples];
    float minimum = surfaceSamplesC[0];
    float maximum = surfaceSamplesC[0];
    for (uint8_t i = 0; i < sampleCount; ++i) {
      const float value = surfaceSamplesC[i];
      if (!isfinite(value) || value < 0.0f || value > 50.0f) {
        return result;
      }
      sorted[i] = value;
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
    }

    insertionSort(sorted, sampleCount);

    result.sampleSpanC = maximum - minimum;
    result.surfaceMedianC = medianOfSorted(sorted, sampleCount);

    if (!thresholdsAreValid()) {
      return result;
    }

    if (result.sampleSpanC > thresholds_.maximumStableSpanC) {
      result.range = SurfaceRange::ContactUnstable;
      return result;
    }
    result.contactStable = true;

    if (!baselineAvailable || !isfinite(healthySleepSurfaceBaselineC)) {
      result.range = SurfaceRange::BaselineRequired;
      return result;
    }

    result.baselineDeltaC =
        result.surfaceMedianC - healthySleepSurfaceBaselineC;
    if (result.baselineDeltaC > thresholds_.highDeltaC) {
      result.range = SurfaceRange::WellAboveBaseline;
      result.severity = SurfaceSeverity::High;
    } else if (result.baselineDeltaC > thresholds_.normalDeltaC) {
      result.range = SurfaceRange::AboveBaseline;
      result.severity = SurfaceSeverity::Caution;
    } else if (result.baselineDeltaC < -thresholds_.highDeltaC) {
      result.range = SurfaceRange::WellBelowBaseline;
      result.severity = SurfaceSeverity::High;
    } else if (result.baselineDeltaC < -thresholds_.normalDeltaC) {
      result.range = SurfaceRange::BelowBaseline;
      result.severity = SurfaceSeverity::Caution;
    } else {
      result.range = SurfaceRange::PersonalBaseline;
      result.severity = SurfaceSeverity::Normal;
    }
    return result;
  }

 private:
  static void insertionSort(float* values, uint8_t count) {
    for (uint8_t i = 1; i < count; ++i) {
      const float value = values[i];
      uint8_t position = i;
      while (position > 0 && values[position - 1] > value) {
        values[position] = values[position - 1];
        --position;
      }
      values[position] = value;
    }
  }

  static float medianOfSorted(const float* values, uint8_t count) {
    return count % 2 == 0
               ? (values[count / 2 - 1] + values[count / 2]) / 2.0f
               : values[count / 2];
  }

  bool configIsValid() const {
    return isfinite(config_.nominalResistanceOhm) &&
           config_.nominalResistanceOhm > 0.0f &&
           isfinite(config_.nominalTemperatureC) &&
           config_.nominalTemperatureC > -273.15f &&
           isfinite(config_.betaK) && config_.betaK > 0.0f &&
           isfinite(config_.fixedResistorOhm) &&
           config_.fixedResistorOhm > 0.0f && config_.adcMaximum > 1;
  }

  bool thresholdsAreValid() const {
    return isfinite(thresholds_.normalDeltaC) &&
           thresholds_.normalDeltaC > 0.0f &&
           isfinite(thresholds_.highDeltaC) &&
           thresholds_.highDeltaC > thresholds_.normalDeltaC &&
           isfinite(thresholds_.maximumStableSpanC) &&
           thresholds_.maximumStableSpanC > 0.0f;
  }

  ThermistorSurfaceConfig config_;
  SurfaceThresholdConfig thresholds_;
};

}  // namespace smart_collar
