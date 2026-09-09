#pragma once

#include <math.h>
#include <stdint.h>

namespace smart_collar {

enum class EnvironmentRange : uint8_t {
  SensorInvalid,
  ExtremeLow,
  VeryLow,
  Low,
  Normal,
  Warm,
  High,
  ExtremeHigh
};

enum class EnvironmentSeverity : uint8_t {
  Normal,
  Caution,
  High,
  Critical,
  Invalid
};

enum class CollarActivity : uint8_t { Resting, Light, Active, Intense };

struct DogThermalProfile {
  bool brachycephalic;
  bool hairless;
  bool singleCoat;
  bool doubleOrHeavyCoat;
  bool toyOrSmall;
  bool giant;
  bool underweight;
  bool overweight;
  bool obese;
  bool vulnerableAge;
  bool respiratoryOrCardiacCondition;
  bool heatAcclimated;
  bool coldAcclimated;
  uint8_t additionalHeatRiskPoints;
  uint8_t additionalColdRiskPoints;

  DogThermalProfile()
      : brachycephalic(false),
        hairless(false),
        singleCoat(false),
        doubleOrHeavyCoat(false),
        toyOrSmall(false),
        giant(false),
        underweight(false),
        overweight(false),
        obese(false),
        vulnerableAge(false),
        respiratoryOrCardiacCondition(false),
        heatAcclimated(false),
        coldAcclimated(false),
        additionalHeatRiskPoints(0),
        additionalColdRiskPoints(0) {}
};

struct EnvironmentContext {
  CollarActivity activity;
  bool directSun;
  bool poorAirflow;
  bool enclosedVehicle;

  EnvironmentContext()
      : activity(CollarActivity::Resting),
        directSun(false),
        poorAirflow(false),
        enclosedVehicle(false) {}
};

struct EnvironmentThresholdConfig {
  float extremeLowC;
  float veryLowC;
  float lowC;
  float warmC;
  float highC;
  float extremeHighC;
  float highHumidityPct;
  float veryHighHumidityPct;
  float filterAlpha;
  float rapidChangeCPerMinute;
  uint32_t rapidChangeMaximumWindowMs;

  EnvironmentThresholdConfig()
      : extremeLowC(-5.0f),
        veryLowC(0.0f),
        lowC(10.0f),
        warmC(25.0f),
        highC(30.0f),
        extremeHighC(35.0f),
        highHumidityPct(65.0f),
        veryHighHumidityPct(80.0f),
        filterAlpha(0.35f),
        rapidChangeCPerMinute(2.0f),
        rapidChangeMaximumWindowMs(120000UL) {}
};

struct EnvironmentAssessment {
  EnvironmentRange range;
  EnvironmentSeverity severity;
  float rawTemperatureC;
  float filteredTemperatureC;
  float relativeHumidityPct;
  bool rapidRiseDetected;
  bool rapidFallDetected;
  bool immediateThresholdBypass;
  uint8_t heatVulnerabilityScore;
  uint8_t coldVulnerabilityScore;
};

class SmartCollarEnvironment {
 public:
  explicit SmartCollarEnvironment(
      const EnvironmentThresholdConfig& thresholds =
          EnvironmentThresholdConfig())
      : thresholds_(thresholds),
        initialized_(false),
        previousRawTemperatureC_(0.0f),
        filteredTemperatureC_(0.0f),
        previousReadingMs_(0) {}

  EnvironmentAssessment update(float temperatureC, float relativeHumidityPct,
                               uint32_t nowMs,
                               const DogThermalProfile& dog,
                               const EnvironmentContext& context) {
    EnvironmentAssessment result = {
        EnvironmentRange::SensorInvalid, EnvironmentSeverity::Invalid,
        temperatureC, 0.0f, relativeHumidityPct, false, false, false, 0, 0};

    if (!thresholdsAreValid() || !isfinite(temperatureC) ||
        !isfinite(relativeHumidityPct) || temperatureC < -40.0f ||
        temperatureC > 125.0f || relativeHumidityPct < 0.0f ||
        relativeHumidityPct > 100.0f) {
      return result;
    }

    bool rapidRise = false;
    bool rapidFall = false;
    if (initialized_) {
      const uint32_t elapsedMs =
          static_cast<uint32_t>(nowMs - previousReadingMs_);
      if (elapsedMs > 0UL &&
          elapsedMs <= thresholds_.rapidChangeMaximumWindowMs) {
        const float rateCPerMinute =
            (temperatureC - previousRawTemperatureC_) * 60000.0f /
            static_cast<float>(elapsedMs);
        rapidRise = rateCPerMinute >= thresholds_.rapidChangeCPerMinute;
        rapidFall = rateCPerMinute <= -thresholds_.rapidChangeCPerMinute;
      }
      filteredTemperatureC_ =
          thresholds_.filterAlpha * temperatureC +
          (1.0f - thresholds_.filterAlpha) * filteredTemperatureC_;
    } else {
      initialized_ = true;
      filteredTemperatureC_ = temperatureC;
    }

    previousRawTemperatureC_ = temperatureC;
    previousReadingMs_ = nowMs;

    // A dangerous absolute reading or sudden change bypasses smoothing so a
    // nearby heat/cold source remains visible rather than being averaged away.
    const bool extremeReading = temperatureC >= thresholds_.extremeHighC ||
                                temperatureC < thresholds_.extremeLowC;
    const bool bypass = extremeReading || rapidRise || rapidFall;
    const float assessedTemperatureC =
        bypass ? temperatureC : filteredTemperatureC_;

    result.rawTemperatureC = temperatureC;
    result.filteredTemperatureC = filteredTemperatureC_;
    result.relativeHumidityPct = relativeHumidityPct;
    result.rapidRiseDetected = rapidRise;
    result.rapidFallDetected = rapidFall;
    result.immediateThresholdBypass = bypass;
    result.range = classifyRange(assessedTemperatureC);
    result.severity = baseSeverity(result.range);
    result.heatVulnerabilityScore = heatScore(dog);
    result.coldVulnerabilityScore = coldScore(dog);

    uint8_t severity = severityValue(result.severity);
    if (assessedTemperatureC >= 20.0f) {
      if (assessedTemperatureC >= 24.0f &&
          relativeHumidityPct >= thresholds_.veryHighHumidityPct) {
        ++severity;
      } else if (assessedTemperatureC >= 24.0f &&
                 relativeHumidityPct >= thresholds_.highHumidityPct) {
        ++severity;
      }
      if (context.activity == CollarActivity::Active) {
        ++severity;
      } else if (context.activity == CollarActivity::Intense) {
        severity = static_cast<uint8_t>(severity + 2U);
      }
      if (context.directSun) ++severity;
      if (context.poorAirflow) ++severity;
      if (result.heatVulnerabilityScore >= 7U) {
        severity = static_cast<uint8_t>(severity + 2U);
      } else if (result.heatVulnerabilityScore >= 3U) {
        ++severity;
      }
      if (rapidRise) ++severity;
      if (context.enclosedVehicle && assessedTemperatureC >= 20.0f) {
        severity = 3U;
      }
    } else if (assessedTemperatureC < thresholds_.lowC) {
      if (result.coldVulnerabilityScore >= 4U) {
        severity = static_cast<uint8_t>(severity + 2U);
      } else if (result.coldVulnerabilityScore >= 1U) {
        ++severity;
      }
      if (rapidFall) ++severity;
    }

    if (severity > 3U) severity = 3U;
    result.severity = severityFromValue(severity);
    return result;
  }

  void reset() {
    initialized_ = false;
    previousRawTemperatureC_ = 0.0f;
    filteredTemperatureC_ = 0.0f;
    previousReadingMs_ = 0;
  }

 private:
  EnvironmentRange classifyRange(float temperatureC) const {
    if (temperatureC < thresholds_.extremeLowC)
      return EnvironmentRange::ExtremeLow;
    if (temperatureC < thresholds_.veryLowC)
      return EnvironmentRange::VeryLow;
    if (temperatureC < thresholds_.lowC) return EnvironmentRange::Low;
    if (temperatureC < thresholds_.warmC) return EnvironmentRange::Normal;
    if (temperatureC < thresholds_.highC) return EnvironmentRange::Warm;
    if (temperatureC < thresholds_.extremeHighC)
      return EnvironmentRange::High;
    return EnvironmentRange::ExtremeHigh;
  }

  static EnvironmentSeverity baseSeverity(EnvironmentRange range) {
    switch (range) {
      case EnvironmentRange::ExtremeLow:
      case EnvironmentRange::ExtremeHigh:
        return EnvironmentSeverity::Critical;
      case EnvironmentRange::VeryLow:
      case EnvironmentRange::High:
        return EnvironmentSeverity::High;
      case EnvironmentRange::Low:
      case EnvironmentRange::Warm:
        return EnvironmentSeverity::Caution;
      case EnvironmentRange::Normal:
        return EnvironmentSeverity::Normal;
      default:
        return EnvironmentSeverity::Invalid;
    }
  }

  static uint8_t severityValue(EnvironmentSeverity severity) {
    switch (severity) {
      case EnvironmentSeverity::Normal:
        return 0U;
      case EnvironmentSeverity::Caution:
        return 1U;
      case EnvironmentSeverity::High:
        return 2U;
      case EnvironmentSeverity::Critical:
        return 3U;
      default:
        return 3U;
    }
  }

  static EnvironmentSeverity severityFromValue(uint8_t value) {
    if (value == 0U) return EnvironmentSeverity::Normal;
    if (value == 1U) return EnvironmentSeverity::Caution;
    if (value == 2U) return EnvironmentSeverity::High;
    return EnvironmentSeverity::Critical;
  }

  static uint8_t heatScore(const DogThermalProfile& dog) {
    int16_t score = dog.additionalHeatRiskPoints;
    if (dog.brachycephalic) score += 3;
    if (dog.doubleOrHeavyCoat) score += 1;
    if (dog.giant) score += 1;
    if (dog.overweight) score += 1;
    if (dog.obese) score += 2;
    if (dog.vulnerableAge) score += 1;
    if (dog.respiratoryOrCardiacCondition) score += 3;
    if (dog.heatAcclimated) --score;
    if (score < 0) score = 0;
    return static_cast<uint8_t>(score);
  }

  static uint8_t coldScore(const DogThermalProfile& dog) {
    int16_t score = dog.additionalColdRiskPoints;
    if (dog.hairless) {
      score += 3;
    } else if (dog.singleCoat) {
      score += 1;
    }
    if (dog.toyOrSmall) score += 2;
    if (dog.underweight) score += 1;
    if (dog.vulnerableAge) score += 1;
    if (dog.doubleOrHeavyCoat) --score;
    if (dog.coldAcclimated) --score;
    if (score < 0) score = 0;
    return static_cast<uint8_t>(score);
  }

  bool thresholdsAreValid() const {
    return thresholds_.extremeLowC < thresholds_.veryLowC &&
           thresholds_.veryLowC < thresholds_.lowC &&
           thresholds_.lowC < thresholds_.warmC &&
           thresholds_.warmC < thresholds_.highC &&
           thresholds_.highC < thresholds_.extremeHighC &&
           thresholds_.highHumidityPct >= 0.0f &&
           thresholds_.veryHighHumidityPct >= thresholds_.highHumidityPct &&
           thresholds_.veryHighHumidityPct <= 100.0f &&
           thresholds_.filterAlpha > 0.0f && thresholds_.filterAlpha <= 1.0f &&
           thresholds_.rapidChangeCPerMinute > 0.0f &&
           thresholds_.rapidChangeMaximumWindowMs > 0UL;
  }

  EnvironmentThresholdConfig thresholds_;
  bool initialized_;
  float previousRawTemperatureC_;
  float filteredTemperatureC_;
  uint32_t previousReadingMs_;
};

}  // namespace smart_collar
