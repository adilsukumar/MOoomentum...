# Frontend value contract

Every `analyze_session(...)` result now contains a compact `frontend` object. The raw
`summary` and `windows` remain available for charts and audit views, but a frontend can
render `frontend.metrics` directly without reproducing medical or engineering logic.

## Shape

```json
{
  "frontend": {
    "schema_version": "vitalsense.frontend.v1",
    "dog_id": "dog-001",
    "reference_profile": {},
    "baseline_training": {},
    "overall": {},
    "metrics": {
      "heart_rate_bpm": {
        "label": "Mechanical heart rate",
        "value": 60.0,
        "unit": "bpm",
        "display_value": "60 bpm",
        "available": true,
        "state": "within_reference",
        "severity": "none",
        "confidence_pct": 94,
        "base": {},
        "thresholds": []
      }
    },
    "groups": {},
    "recovery": {},
    "alerts": []
  }
}
```

Use `value` for calculations and `display_value` only for display. `available=false`
and `display_value="—"` means the quality/context gate did not produce a usable value.

`state` compares a value with its descriptive reference or reports signal state.
`severity` comes only from a persistence-aware alert rule. A value can therefore be
`above_reference` while its alert severity is still `none`.

## Base values sent to the frontend

The engine selects an age/weight population IQR until an eligible personal baseline is
available. These are descriptive values from the AI-COLLAR 2025 cohort, not diagnostic
limits.

| Dog profile | HR center (IQR), bpm | RR center (IQR), breaths/min |
|---|---:|---:|
| Puppy, <=12 months | 78.5 (69.1–96.8) | 20.1 (16.0–25.7) |
| Adult, <=10 kg | 65.0 (60.9–69.4) | 17.2 (15.0–19.9) |
| Adult, >10 kg | 59.5 (54.6–64.2) | 15.8 (13.7–18.3) |
| Adult, weight unavailable | 60.5 (55.2–65.3) | 15.7 (13.1–18.7) |
| Senior, >=10 years | 68.7 (60.8–77.2) | 15.7 (13.1–18.7) |

When age is unavailable, the adult row is explicitly marked as a frontend fallback.
The profile should be completed before using population comparisons.

An eligible personal baseline replaces the HR/RR population base. Its displayed band is
the dog's median plus or minus three robust MAD scales. Eligibility defaults to at least
7 days and 100 eligible samples. Mean beat and breath interval bases are calculated as
the inverse of the selected HR/RR base.

There is no validated universal canine normal range in this project for mechanical
RMSSD, SDNN-like SD, pNN50, Poincare metrics, respiratory variability, coupling,
sleep fragmentation, activity, recovery, or relative sensor amplitudes. Their `base.kind`
is `personal_baseline_required`; the frontend must not label them normal or abnormal.

## Alert guardrails sent with HR and RR

| Metric | Limit | Severity | Persistence |
|---|---:|---|---|
| HR | <30 bpm | urgent | 30 continuous seconds, high quality |
| HR | <40 bpm | high | 3 of 5 valid 60-second windows |
| HR | >140 bpm | high | 3 of 5 valid 60-second windows |
| HR | >180 bpm | urgent | 30 continuous seconds, high quality |
| Sleeping RR | <8 breaths/min | info | 2 valid 60-second windows; first check signal/contact |
| Sleeping RR | >25 breaths/min | watch | repeated sleeping readings / 2 nights |
| Resting RR | >=30 breaths/min | high | 2 valid windows separated by 10 minutes |
| Resting RR | >=40 breaths/min | urgent | 5 continuous valid minutes |

Symptoms always override the number. These are conservative research/product guardrails
and require prospective validation on final VitalSense hardware.

## Metric IDs

The groups and stable IDs include:

- `vitals`: mechanical HR and fused RR;
- `pulse_variability`: mean IBI, SD, RMSSD, pNN50, CV, Poincare SD1/SD2;
- `respiratory_variability`: mean breath interval, SD, RMSSD, CV, irregularity;
- `coupling`: beats per breath and phase-locking value;
- `sleep`: time, efficiency, bouts, longest bout, fragmentation and quiet rest;
- `activity`: active minutes, ENMO, bouts and current context;
- `posture`: position changes, orientation transitions and current axis bin;
- `recovery`: starting HR/RR and 1/3/5/10-minute HR/RR drops;
- `signal`: amplitudes, confidence, coverage, contact loss, dog-motion artifacts,
  Piezo press/impact artifacts and respiratory-dropout candidates.

A Piezo press is treated as a large impulse artifact candidate. Its window is invalidated,
reported in `piezo_impulse_artifact_window_pct`, and excluded from HR/RR and alert logic.

