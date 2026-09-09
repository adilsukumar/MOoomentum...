# VitalSense AI

Research foundation for canine vital-sign monitoring with a Piezo/BCG sensor and Bosch BMI270 IMU.

## Start here

- [Vital-sign engine specification](docs/vitals_engine_spec.md) — sensor roles, processing, personal baselines, alerts, derived metrics, validation, and safety limits.
- [Population reference ranges](data/reference/canine_resting_reference_ranges.csv) — age, size, circadian, healthy, and cardiac cohorts.
- [Breed reference table](data/reference/canine_breed_resting_reference.csv) — the seven adequately represented breeds in the 2025 AI-COLLAR cohort.
- [Initial alert configuration](config/vital_alert_thresholds.csv) — machine-readable, deliberately conservative proposal requiring prospective validation.
- [VitalSense dataset dictionary](data/schema/vitalsense_dataset_dictionary.csv) — raw sensors, reference labels, metadata, annotations, and derived fields to collect.
- [Open canine vital dataset](data/public/invoxia_dog_health_vitals/README.md) — 40 dogs, 93 hours, ECG waveforms, beat annotations, and video-derived breathing labels.
- [Dataset summary script](scripts/summarize_public_dataset.py) — reproducible inspection of the public CSV.
- [Executable analytics implementation](docs/implementation.md) — input contract, CLI, calculated outputs, baseline workflow, and tests.
- [Frontend value contract](docs/frontend_contract.md) — stable metric IDs, display values, selected bases, reference ranges, alert limits, confidence and unavailable states.
- `src/vitalsense` — Python package implementing the complete research pipeline.
- [Synthetic end-to-end analysis](examples/synthetic_analysis.json) — known-rate demonstration with activity, recovery, and a posture change.

## Important limitation

Piezo + BMI270 can support resting mechanical pulse, respiratory rate, activity/rest context, variability, and trend detection after device-specific validation. They cannot by themselves diagnose arrhythmia, heart failure, sleep apnea, blood pressure, oxygen saturation, or core temperature.
