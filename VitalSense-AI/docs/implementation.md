# VitalSense analytics implementation

This repository contains an executable research pipeline for synchronized Piezo and BMI270 samples.

## Input contract

CSV columns:

```text
timestamp_s,piezo,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps
```

`timestamp_ns` may replace `timestamp_s`. Gyroscope columns default to zero if omitted; all other channels are required. Samples must be synchronized, strictly increasing, and approximately uniform.

## Run the synthetic end-to-end demo

```powershell
$env:PYTHONPATH='D:\25_VitalSense-AI\src'
python -m vitalsense demo `
  --sensor-output 'D:\25_VitalSense-AI\examples\synthetic_session.csv' `
  --analysis-output 'D:\25_VitalSense-AI\examples\synthetic_analysis.json'
```

The demo contains known 72 bpm mechanical pulse, 15 breaths/min respiration, a motion bout, recovery, and a posture change.

## Analyze device data

```powershell
$env:PYTHONPATH='D:\25_VitalSense-AI\src'
python -m vitalsense analyze raw_session.csv `
  --profile examples/dog_profile.json `
  --config config/analysis.default.json `
  --output analysis.json
```

After `python -m pip install -e .`, the shorter `vitalsense analyze ...` command is available.

## Build a personal baseline

Supply JSON/JSONL rows with `date`, `context`, `heart_rate_bpm`, and `respiratory_rate_bpm`:

```powershell
python -m vitalsense baseline history.jsonl `
  --dog-id dog-001 `
  --output dog-001-baseline.json
```

Then add `--baseline dog-001-baseline.json` to `analyze`.

For persistent multi-day change detection:

```powershell
python -m vitalsense trend history.jsonl `
  --baseline dog-001-baseline.json `
  --output dog-001-trend.json
```

The default baseline gate requires at least seven distinct days and 100 eligible samples. Trend output includes daily medians, rolling three-day medians, robust z scores, relative changes, and persistent-shift alerts.

## Implemented outputs

- quality-gated mechanical pulse and respiratory event times;
- pulse rate, respiratory rate, beat intervals, and breath intervals;
- mechanical inter-beat SD, RMSSD, pNN50, CV, Poincare SD1/SD2;
- breath-interval SD, RMSSD, CV, and irregularity index;
- beats per breath and respiratory phase-locking value;
- ENMO/activity intensity, static/moving context, sustained sleep candidate;
- orientation bins, position changes, active/rest/sleep bouts and fragmentation;
- post-activity 1/3/5/10-minute recovery summaries;
- pulse/respiratory amplitude, artifact/contact burden and usable coverage;
- panting/high-frequency-breathing rejection and possible respiratory-pause-or-signal-loss candidates;
- robust personal-baseline z scores, relative changes and anomaly score;
- multi-day rolling medians and persistent 20% baseline-shift alerts;
- conservative HR/RR/irregular-pulse candidate alerts with messages and rule IDs.
- a stable `frontend` view model containing values, display strings, units, availability,
  confidence, selected personal/population bases, reference comparison, severity and
  threshold persistence for every supported factor.

See [the frontend value contract](frontend_contract.md) for the exact base values and
metric IDs. A large Piezo press/impact is marked `piezo_impulse_artifact_candidate`; the
affected window is invalidated and its burden is exposed to the frontend instead of being
reported as a heart-rate spike.

Every 60-second result includes `valid`, modality/fusion quality, and `reason_codes`. Invalid windows remain in the output for auditability but are excluded from physiological summaries.

### Exact output map

| Requested factor | Analysis JSON output |
|---|---|
| Mechanical pulse rate | `windows[].heart_rate_bpm`, `summary.heart_rate_bpm` |
| Respiratory rate | `windows[].respiratory_rate_bpm`, `summary.respiratory_rate_bpm` |
| Beat/breath event timing | `windows[].pulse_intervals_s`, `windows[].breath_intervals_s` |
| RMSSD, SD, pNN50, Poincare | `windows[].interbeat_metrics`, `summary.aggregate_interbeat_metrics` |
| Respiratory variability | `windows[].respiratory_variability`, `summary.aggregate_respiratory_variability` |
| Rest/sleep/fragmentation | `windows[].context`, `summary.sleep_candidate_*`, `summary.sleep_fragmentation_*` |
| Posture/position changes | `windows[].orientation_bin`, `summary.position_change_count` |
| Activity and active minutes | `windows[].activity_*`, `summary.active_minutes`, `summary.activity_enmo_mg` |
| Post-activity recovery | `summary.recovery_events[].measurements.minute_1/3/5/10` |
| Cardiorespiratory coupling | `windows[].cardiorespiratory_coupling`, aggregate equivalent |
| Personal anomaly score | `personal_baseline.metrics`, `personal_baseline.overall_anomaly_score` |
| Multi-day trends | output of `vitalsense trend`: `daily_medians`, `rolling_3day`, `alerts` |
| Signal confidence | `windows[].piezo_quality`, `imu_quality`, `fusion_quality` |
| Usable coverage | `summary.usable_data_coverage_pct`, `valid_window_count` |
| Additional implemented factors | pulse/breath amplitude, panting candidate, respiratory pause-or-signal-loss candidate, artifact burden, contact loss, rate percentiles, sleep bouts and activity bouts |

## Interpretation boundaries

- `sleep_candidate` means sustained BMI270 immobility; it is not EEG-confirmed sleep.
- inter-beat metrics are mechanical variability, not validated ECG HRV.
- irregular timing is an irregular-pulse candidate, not an arrhythmia diagnosis.
- apparent apnea may be collar contact loss; this hardware does not measure airflow or SpO2.
- orientation bins (`+x`, `-x`, etc.) require a collar-mount calibration table before they can be named left/right/sternal/upright postures.
- detector bands, quality cutoffs, alert persistence and activity thresholds are starting configurations requiring final-hardware validation.

## Test suite

```powershell
$env:PYTHONPATH='D:\25_VitalSense-AI\src'
python -m unittest discover -s tests -v
```
