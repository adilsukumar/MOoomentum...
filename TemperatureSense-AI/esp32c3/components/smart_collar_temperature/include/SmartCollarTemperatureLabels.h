#pragma once

#include "SmartCollarEnvironment.h"
#include "SmartCollarSurfaceTemperature.h"

namespace smart_collar {

inline const char* environmentRangeLabel(EnvironmentRange range) {
  switch (range) {
    case EnvironmentRange::ExtremeLow:
      return "EXTREME_LOW";
    case EnvironmentRange::VeryLow:
      return "VERY_LOW";
    case EnvironmentRange::Low:
      return "LOW";
    case EnvironmentRange::Normal:
      return "NORMAL";
    case EnvironmentRange::Warm:
      return "WARM";
    case EnvironmentRange::High:
      return "HIGH";
    case EnvironmentRange::ExtremeHigh:
      return "EXTREME_HIGH";
    default:
      return "SENSOR_INVALID";
  }
}

inline const char* environmentSeverityLabel(EnvironmentSeverity severity) {
  switch (severity) {
    case EnvironmentSeverity::Normal:
      return "NORMAL";
    case EnvironmentSeverity::Caution:
      return "CAUTION";
    case EnvironmentSeverity::High:
      return "HIGH";
    case EnvironmentSeverity::Critical:
      return "CRITICAL";
    default:
      return "INVALID";
  }
}

inline const char* environmentActionLabel(EnvironmentSeverity severity) {
  switch (severity) {
    case EnvironmentSeverity::Normal:
      return "CONTINUE_MONITORING";
    case EnvironmentSeverity::Caution:
      return "REDUCE_EXPOSURE_AND_CHECK_DOG";
    case EnvironmentSeverity::High:
      return "STOP_EXPOSURE_MOVE_TO_SAFER_AREA";
    case EnvironmentSeverity::Critical:
      return "ACT_NOW_COOL_OR_WARM_AND_SEEK_EMERGENCY_VET_IF_SIGNS";
    default:
      return "CHECK_SENSOR";
  }
}

inline const char* surfaceRangeLabel(SurfaceRange range) {
  switch (range) {
    case SurfaceRange::BaselineRequired:
      return "BASELINE_REQUIRED";
    case SurfaceRange::TooFewSamples:
      return "TOO_FEW_SAMPLES";
    case SurfaceRange::SensorInvalid:
      return "SENSOR_INVALID";
    case SurfaceRange::ContactUnstable:
      return "CONTACT_UNSTABLE";
    case SurfaceRange::WellBelowBaseline:
      return "WELL_BELOW_BASELINE";
    case SurfaceRange::BelowBaseline:
      return "BELOW_BASELINE";
    case SurfaceRange::PersonalBaseline:
      return "NORMAL_PERSONAL_BASELINE";
    case SurfaceRange::AboveBaseline:
      return "ABOVE_BASELINE";
    case SurfaceRange::WellAboveBaseline:
      return "WELL_ABOVE_BASELINE";
    default:
      return "SENSOR_INVALID";
  }
}

inline const char* surfaceSeverityLabel(SurfaceSeverity severity) {
  switch (severity) {
    case SurfaceSeverity::Normal:
      return "NORMAL";
    case SurfaceSeverity::Caution:
      return "CAUTION";
    case SurfaceSeverity::High:
      return "HIGH";
    default:
      return "INVALID";
  }
}

inline const char* surfaceActionLabel(SurfaceSeverity severity) {
  switch (severity) {
    case SurfaceSeverity::Normal:
      return "CONTINUE_MONITORING";
    case SurfaceSeverity::Caution:
      return "RECHECK_SURFACE_CONTACT_AND_CHECK_DOG";
    case SurfaceSeverity::High:
      return "CHECK_DOG_CONTACT_VET_IF_PERSISTENT_OR_SYMPTOMATIC";
    default:
      return "CHECK_SENSOR_CONTACT_OR_BASELINE";
  }
}

}  // namespace smart_collar
