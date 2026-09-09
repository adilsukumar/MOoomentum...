#pragma once

#include <stdint.h>

namespace smart_collar {

struct TemperatureGateConfig {
  uint32_t awakeEnvironmentIntervalMs;
  uint32_t sleepingSurfaceIntervalMs;
  uint32_t sleepConfirmationMs;
  // Set to zero to stop SHT40 measurements during confirmed sleep.
  uint32_t sleepingEnvironmentIntervalMs;

  TemperatureGateConfig()
      : awakeEnvironmentIntervalMs(30000UL),
        sleepingSurfaceIntervalMs(60000UL),
        sleepConfirmationMs(300000UL),
        sleepingEnvironmentIntervalMs(0UL) {}
};

struct TemperatureGateDecision {
  bool readSHT40;
  bool readNTC;
  bool sleepCandidate;
  bool sleepConfirmed;
};

class SmartCollarTemperatureGate {
 public:
  explicit SmartCollarTemperatureGate(
      const TemperatureGateConfig& config = TemperatureGateConfig())
      : config_(config),
        initialized_(false),
        modelSaysSleeping_(false),
        stateSinceMs_(0),
        hasSHT40Reading_(false),
        hasNTCReading_(false),
        lastSHT40Ms_(0),
        lastNTCMs_(0) {}

  // `modelSaysSleeping` is the final boolean produced by the existing AI code.
  // Call update once after every inference and regularly from loop().
  TemperatureGateDecision update(bool modelSaysSleeping, uint32_t nowMs) {
    if (!initialized_) {
      initialized_ = true;
      modelSaysSleeping_ = modelSaysSleeping;
      stateSinceMs_ = nowMs;
    } else if (modelSaysSleeping != modelSaysSleeping_) {
      modelSaysSleeping_ = modelSaysSleeping;
      stateSinceMs_ = nowMs;
    }

    const bool sleepConfirmed =
        modelSaysSleeping_ &&
        elapsedAtLeast(nowMs, stateSinceMs_, config_.sleepConfirmationMs);

    TemperatureGateDecision decision = {false, false,
                                        modelSaysSleeping_, sleepConfirmed};

    if (sleepConfirmed) {
      decision.readNTC = isDue(nowMs, lastNTCMs_, hasNTCReading_,
                               config_.sleepingSurfaceIntervalMs);
      if (config_.sleepingEnvironmentIntervalMs > 0UL) {
        decision.readSHT40 =
            isDue(nowMs, lastSHT40Ms_, hasSHT40Reading_,
                  config_.sleepingEnvironmentIntervalMs);
      }
      return decision;
    }

    // Awake, uncertain, and not-yet-confirmed sleep all keep the NTC off.
    decision.readSHT40 = isDue(nowMs, lastSHT40Ms_, hasSHT40Reading_,
                               config_.awakeEnvironmentIntervalMs);
    return decision;
  }

  // Call these only after a successful sensor measurement. If a read fails,
  // do not mark it; the gate will request it again on the next loop.
  void markSHT40Read(uint32_t nowMs) {
    hasSHT40Reading_ = true;
    lastSHT40Ms_ = nowMs;
  }

  void markNTCRead(uint32_t nowMs) {
    hasNTCReading_ = true;
    lastNTCMs_ = nowMs;
  }

  void reset() {
    initialized_ = false;
    modelSaysSleeping_ = false;
    stateSinceMs_ = 0;
    hasSHT40Reading_ = false;
    hasNTCReading_ = false;
    lastSHT40Ms_ = 0;
    lastNTCMs_ = 0;
  }

 private:
  static bool elapsedAtLeast(uint32_t nowMs, uint32_t sinceMs,
                             uint32_t intervalMs) {
    // Unsigned subtraction remains correct across uint32_t timer rollover.
    return static_cast<uint32_t>(nowMs - sinceMs) >= intervalMs;
  }

  static bool isDue(uint32_t nowMs, uint32_t lastMs, bool hasReading,
                    uint32_t intervalMs) {
    return !hasReading || elapsedAtLeast(nowMs, lastMs, intervalMs);
  }

  TemperatureGateConfig config_;
  bool initialized_;
  bool modelSaysSleeping_;
  uint32_t stateSinceMs_;
  bool hasSHT40Reading_;
  bool hasNTCReading_;
  uint32_t lastSHT40Ms_;
  uint32_t lastNTCMs_;
};

}  // namespace smart_collar
