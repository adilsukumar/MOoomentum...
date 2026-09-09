# VitalSense AI canine vital-sign engine

Version: 0.1 research specification (23 August 2026)

## Bottom line

Yes, each dog has a different resting pulse and breathing baseline. Age, body size, breed, time of day, season, posture, sleep/wake state, fitness, temperature, stress, medication, and disease all matter. VitalSense should therefore combine:

1. **Population guardrails** — broad, evidence-backed limits that prevent obviously implausible or concerning values from being treated as normal.
2. **A context-specific personal baseline** — learned separately for sleep and quiet awake rest.
3. **Signal-quality and persistence rules** — never alert from one noisy window.

The wearable is a trend and screening system. It must not diagnose arrhythmia, heart failure, sleep apnea, or another disease from Piezo + BMI270 alone.

## What the evidence says

The 2025 AI-COLLAR prospective study followed 703 apparently healthy dogs in their homes. Its adult, non-senior cohort had a median resting HR of 60.5 bpm (IQR 55.2–65.3) and RR of 16.1 breaths/min (IQR 13.8–18.7). Night values were lower than day values. Puppies and senior dogs had higher median HR than adults, and dogs at or below 10 kg had higher HR than heavier dogs. Seven sufficiently represented breeds also showed modest differences. These are descriptive population values, not diagnostic cutoffs. [Chetboul et al., 2025](https://doi.org/10.3389/fvets.2025.1667355)

In a separate home study of 114 apparently healthy adult dogs, mean sleeping RR was 13 breaths/min, no dog's mean was above 23, and instantaneous readings above 30 were rare. [Rishniw et al., 2012](https://doi.org/10.1016/j.rvsc.2011.12.014)

Piezo/ballistocardiography is technically credible at rest, but it is not ECG. A 12-dog BCG study found strong HR agreement with ECG but weaker individual RR agreement; awake-dog RR limits of agreement were roughly -4.2 to +4.9 breaths/min. [Chuluunbaatar et al., 2025](https://doi.org/10.3390/vetsci12040301)

An independent 2025 harness study reinforces the product rule: wearable trends can be useful, but individual readings should be manually confirmed before clinical decisions. [Validation of a wireless harness in hospitalized dogs](https://doi.org/10.3390/vetsci12070626)

## Sensor roles

| Sensor | Primary jobs | Secondary jobs | Cannot establish |
|---|---|---|---|
| Piezo/BCG | mechanical pulse times, pulse rate, cardiac vibration envelope | breathing waveform, inter-beat variability, contact quality | ECG rhythm, electrical conduction, blood pressure, oxygen saturation |
| BMI270 accelerometer + gyroscope | motion rejection, static/recumbent state, posture candidate, respiratory micromotion | activity, rest fragmentation, position changes | true sleep stage, airflow, oxygenation, core temperature |
| Fusion | confidence scoring, artifact removal, agreement checks | recovery trends, cardiorespiratory coupling | a medical diagnosis |

Static is not synonymous with asleep. Call the output `sleep_candidate` until it has been validated against synchronized video and, for sleep-stage claims, polysomnography.

## Starting acquisition configuration

These are engineering starting points, not validated final settings.

- Use one monotonic hardware clock and timestamp Piezo and BMI270 data together.
- Piezo ADC: start at 200 Hz or higher, with adequate analog anti-alias filtering.
- BMI270 static vital mode: accelerometer 100 Hz, +/-2 g, performance filtering; gyroscope 100–200 Hz at +/-125 or +/-250 dps when battery permits.
- BMI270 activity-only mode can run at 25–50 Hz. Bosch documents the available ranges, ODRs, FIFO, and synchronization facilities in the [BMI270 datasheet](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf).
- Begin a vital window only after at least 40 seconds of lying/immobility; this matches the successful real-world collar protocol, but must be revalidated for the VitalSense enclosure and mounting.
- Store raw data. Do not retain only calculated bpm values; raw signals are required for debugging, re-labeling, and future algorithms.

Candidate signal bands:

- Respiration: 0.10–0.80 Hz (6–48 breaths/min) while non-panting.
- Cardiac mechanical component: approximately 0.30–4.0 Hz (18–240 bpm), with morphology/harmonics used by the detector. The low end exists for bradycardia screening and overlaps respiration, so morphology and multimodal validation are essential.
- Movement/artifact: detected from acceleration/gyro energy, clipping, sudden orientation changes, and Piezo saturation.

Because the respiratory and cardiac bands can overlap at the edge, use adaptive decomposition or model-based fusion rather than two fixed filters alone.

## Processing pipeline

1. Calibrate BMI270 offsets and verify Piezo contact after attachment.
2. Classify context: `moving`, `static_upright`, `recumbent_awake_candidate`, `sleep_candidate`, `panting_or_unusable`.
3. Calculate a 0–1 signal-quality score for Piezo, accelerometer respiration, and fusion.
4. Reject clipping, collar adjustment, scratching, grooming, dream twitches, posture transitions, and panting.
5. Detect mechanical pulse candidates from Piezo; use IMU as an artifact reference.
6. Detect breath cycles from the best low-frequency axis/component; compare against Piezo respiration.
7. Calculate window metrics only when quality and duration pass.
8. Aggregate by context into 1-minute, hourly, sleep-session, and daily summaries.
9. Run population guardrails and personal change detection.
10. Emit the measurement, confidence, reason codes, and recommended confirmation—not a diagnosis.

## Personal baseline

Create separate baselines for `sleep_candidate` and `quiet_awake_rest`.

Baseline eligibility:

- dog profile complete;
- no owner-reported illness or medication change;
- at least 7 days, preferably 14–30 days;
- at least 100 valid one-minute windows distributed across at least 7 days;
- no panting and fusion quality >= 0.80;
- baseline freezes during illness, travel, unusual heat, post-operative periods, or medication changes.

For metric `x` in one context:

```text
center = median(x over rolling 28-day healthy baseline)
scale  = 1.4826 * median(abs(x - center))       # robust MAD scale
robust_z = (x - center) / max(scale, minimum_scale)
relative_change = (current_3day_median - center) / center
```

Suggested anomaly logic:

- short event: `abs(robust_z) >= 3` in at least 3 of 5 valid windows;
- slow trend: absolute 3-day median change >=20% from the 28-day baseline;
- high-priority trend: the shift persists 3 days or crosses a population safety guardrail;
- reset or adapt the baseline slowly; never let an ongoing abnormal rise immediately redefine “normal.”

The 20%/3-day rule is a conservative engineering proposal that requires prospective validation. The AI-COLLAR report supports long-moving-baseline detection, but does not publish a universal alert threshold.

## Initial alert policy

The machine-readable proposal is in `config/vital_alert_thresholds.csv`. Important interpretation:

- **RR >=30 during true sleep/rest:** ask the owner to visually recount for a full minute. Repeated sleeping RR at or above 30 or a sustained rise above personal baseline warrants veterinary contact, especially in a dog with known heart disease.
- **RR >=40 at rest or any labored breathing:** high priority. Visible effort, blue/pale gums, collapse, inability to settle, or distress overrides all numeric rules and requires urgent veterinary care.
- **Low RR:** published stable dogs can have sleeping rates around 7. A low number alone has no well-validated universal disease cutoff. Report `possible_apnea_or_signal_loss`, seek confirmation, and prioritize repeated cessation/effort over bpm.
- **HR:** do not alert merely because sleeping HR is below 60; that is common in home resting data. Use personal deviation first. Sustained values below 40 or above 140 while truly static deserve confirmation; more extreme sustained values receive higher priority.
- A mechanical collar cannot diagnose an arrhythmia. Irregular timing is `irregular_pulse_candidate` and should be confirmed with ECG/Holter.

Published oncology adverse-event criteria list canine sinus bradycardia bands below 60/40/30/20 bpm and sinus tachycardia bands above 140/180/200/240 bpm, but those criteria are not a consumer-wearable alarm standard and require ECG/context. They are used here only as outer guardrails. [VCOG-CTCAE v2](https://doi.org/10.1111/jvim.16138)

## Metrics VitalSense can calculate

### Reasonable with these sensors after validation

- pulse rate / mechanical HR;
- respiratory rate during clean static windows;
- pulse and breath interval series;
- resting/sleep personal baselines and circadian difference;
- inter-beat variability: SD of intervals, RMSSD, pNN50, Poincare SD1/SD2;
- respiratory interval variability and irregular-breathing candidates;
- beats-per-breath and exploratory cardiorespiratory coupling;
- recumbent/static time, sleep-candidate duration, rest fragmentation, position changes;
- activity intensity, active minutes, and post-activity HR/RR recovery;
- trend scores, anomaly scores, measurement confidence, and data coverage.

Call the variability output `mechanical inter-beat variability`, not clinical HRV, until paired ECG shows that missed/extra mechanical peaks and ectopy do not bias it. Compute 5-minute metrics only on high-quality, artifact-free windows and always retain coverage and rejected-beat percentages.

### Possible only with additional sensors or reference data

- core/body temperature: add a validated temperature channel; collar surface temperature is not core temperature;
- ambient heat/humidity correction: add environmental temperature and humidity;
- oxygen saturation: requires a validated optical or other oxygenation sensor;
- ECG rhythm and arrhythmia diagnosis: requires ECG/Holter;
- blood pressure, stroke volume, cardiac output: not derivable from Piezo + BMI270 without a separately validated method/reference;
- calories: only a model-based estimate using weight, activity, age, neuter status, and validation calorimetry—not a direct measurement;
- clinical sleep stages or sleep apnea: requires airflow/effort, oxygenation, and usually EEG for validation.

## Dataset plan

### Public data now in this repository

`data/public/invoxia_dog_health_vitals` contains the CC BY 4.0 Dog Health Vitals Dataset:

- 40 dogs with varied breed, size, age, and coat;
- about 5,585 minutes (93 hours) of home sleeping sessions;
- 120 Hz portable ECG waveforms;
- expert-reviewed ECG pulse times;
- video-derived breathing-rate segments;
- session/dog metadata and poor-ECG intervals.

It is excellent for validating labels and metric code. It does **not** contain your Piezo or BMI270 channels, so it cannot train the final VitalSense sensor-to-vitals model.

PhysioZoo is a second useful open reference: 17 conscious dogs, 500 Hz ECG, corrected R peaks, and signal-quality annotations. It is useful for HRV code, not for respiration or sensor fusion. [PhysioZoo on PhysioNet](https://doi.org/10.13026/p63q-hq95)

### VitalSense data that must be collected

Each labeled recording must synchronize:

- raw Piezo ADC;
- BMI270 accel x/y/z, gyro x/y/z, sensor timestamps and configuration;
- reference ECG with expert-reviewed R peaks;
- respiration reference (respiratory inductance belt and synchronized thoracic video/manual annotation);
- synchronized behavior/posture/sleep video;
- dog profile and session context;
- attachment position, collar tension, coat, chest/neck conformation, firmware, enclosure, and battery state;
- signal-quality and artifact annotations;
- veterinary health status and medication, when consented.

The exact field dictionary is in `data/schema/vitalsense_dataset_dictionary.csv`.

### Recommended cohort (product-development target, not a regulatory sample-size rule)

- at least 150 healthy dogs: >=30 puppies, >=80 adults, >=40 seniors;
- balance weight bins `<=10`, `>10–20`, `>20–35`, `>35 kg`;
- deliberately include short/long/thick coats, brachycephalic dogs, deep/barrel/keel chest types, and mixed breeds;
- at least 2 hours of synchronized gold-standard rest/sleep per dog across positions, plus 14 days of home longitudinal data;
- include quiet awake rest, natural sleep, post-activity recovery, mild household motion/artifacts, and warm conditions without inducing unsafe stress;
- build a separate, veterinarian-supervised disease/medication cohort for alert validation; do not use healthy-dog thresholds as a substitute.

Split train/validation/test **by dog**, never by time window. Otherwise adjacent windows from the same dog leak identity and waveform morphology into both training and test sets. Keep a locked external test set covering hardware units, homes, breeds, weights, coats, and attachment orientations not used for tuning.

## Minimum validation gates before product alerts

- HR and RR error reported as MAE, RMSE, bias, and Bland–Altman limits of agreement—not correlation alone.
- Beat detection precision/recall/F1 at +/-50 ms and +/-100 ms.
- Respiratory cycle detection precision/recall plus RR error.
- Sensitivity/specificity and false alerts per dog-day for every alert tier.
- Results stratified by dog, age, weight, breed group, coat, conformation, posture, time of day, attachment, and signal quality.
- Coverage: percentage of wear time that yields a valid metric.
- Reproducibility after collar removal/re-attachment.
- Manual confirmation workflow tested with owners and veterinarians.
- Prospective external validation on dogs and clinics not involved in model development.

## Safety language for the app

Use: “VitalSense detected a sustained change from your dog’s usual sleeping breathing rate. Recount breaths for one full minute while your dog sleeps. Contact your veterinarian if the change is confirmed or your dog has symptoms.”

Avoid: “Your dog has heart failure,” “arrhythmia detected,” “sleep apnea diagnosed,” or “medical-grade” until the exact VitalSense hardware, algorithms, intended use, and clinical validation support those claims.

## Primary references

1. [Chetboul et al. (2025), AI-COLLAR prospective cohort](https://doi.org/10.3389/fvets.2025.1667355)
2. [Rishniw et al. (2012), sleeping RR in apparently healthy dogs](https://doi.org/10.1016/j.rvsc.2011.12.014)
3. [Ljungvall et al. (2013), sleeping/resting RR in subclinical heart disease](https://doi.org/10.2460/javma.243.6.839)
4. [Porciello et al. (2016), RR in medically controlled CHF](https://doi.org/10.1016/j.tvjl.2015.08.017)
5. [Chuluunbaatar et al. (2025), canine wearable BCG versus ECG](https://doi.org/10.3390/vetsci12040301)
6. [Jarkoff et al. (2023), smart-collar validation and public dataset](https://doi.org/10.1101/2023.06.09.544347)
7. [Bálint et al. (2019), ECG and respiration across canine sleep phases](https://doi.org/10.3389/fnbeh.2019.00207)
8. [Bosch Sensortec BMI270 datasheet](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf)
9. [VCOG-CTCAE v2](https://doi.org/10.1111/jvim.16138)
10. [Keene et al. (2019), ACVIM MMVD consensus](https://doi.org/10.1111/jvim.15488)
