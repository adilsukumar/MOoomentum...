# Goat MotionSense AI

Pretrained accelerometer pipeline for four mutually exclusive behaviours—**Grazing,
Resting, Walking, Rumination**—plus a separate **Normal gait / Mobility anomaly**
status while the goat is walking.

The two outputs are deliberately hierarchical: lameness is not a behaviour class.
The mobility result is an anomaly screen and **must not be presented as a foot-rot
diagnosis**.

## Data decision

The supplied [Zenodo record 17853479](https://doi.org/10.5281/zenodo.17853479)
contains useful raw goat acceleration and labels including eating (`Comiendo`),
walking/moving, rumination, inactivity and lying. It can supervise a separate
neck-mounted behaviour expert, although eating is only a proxy for pasture grazing
and it has no veterinary lameness labels.

The pretrained behaviour model therefore uses
[CabriTrack](https://doi.org/10.57745/CYF9QW): 144+ annotated hours, 59 goats,
25 Hz triaxial accelerometers, and labels for grazing, resting,
ruminating/chewing and displacement. `Displacement` is mapped to `Walking`, with
the documented limitation that it also includes running and posture transitions.
Validation is split by animal, not by randomly mixed windows.

The expanded model also uses one raw recording from each of eight dairy goats in
[MoSAR](https://doi.org/10.57745/LGZBM1). Its 5 Hz ear-tag signals are resampled to
the common feature rate. Explicit walking and rumination labels are retained;
clean lying/standing periods become Resting. Indoor `feeder` activity is excluded
rather than mislabeled as pasture Grazing.

Because horn and ear mounting produce materially different signals, the bundle
contains separate experts. The production `horn` profile supports all four target
behaviours. The `ear`/MoSAR expert is retained as experimental evidence only: its
held-out-animal result is poor and it has no pasture-grazing label, so it must not
be substituted for the horn profile merely to claim more training data.

The full supplied UPV/Zenodo archive is also supported as a separate neck profile.
Its actual labels cover eating, walking/moving, inactivity/lying and rumination.
`Comiendo` is a feeding proxy rather than a pasture-specific grazing observation,
so its profile must not be used to claim field-grazing validation.

The included gait model is an Isolation Forest fitted only to normal goat walking.
It flags unusual walking windows; useful as an early-warning screen, but its
sensitivity/specificity for foot rot is unknown until locally validated. A useful
cross-species reference is the CC0
[Dryad sheep-lameness dataset](https://doi.org/10.5061/dryad.mk4fc3r), which has 23
sheep, ear accelerometer + gyroscope features and sound/lame labels. It was not
silently merged: it contains precomputed 7-second features rather than raw signals,
uses sheep at 16 Hz, and reports only 76.83% accuracy for walking. Treating that as
ground truth for goats would make the result look stronger than it is.

## Sensor and CSV contract

Mount the BMI270 consistently (collar/horn/leg placement changes the signal). The
public pretraining source uses a horn-mounted accelerometer. The model runs at
25 Hz and expects acceleration in **g**, with columns:

```text
acc_x,acc_y,acc_z
```

Aliases `x/y/z` and `ax/ay/az` are accepted. The BMI270 gyroscope and piezo are
valuable, but the public pretraining data do not contain them. To benefit from
those channels, collect locally labeled data and retrain/refine after extending
the feature configuration; do not silently mix their values into an
accelerometer-only model.

## Train

```powershell
python -m pip install -r requirements.txt
python train.py --cabritrack data/raw/cabritrack.txt --mosar-dir data/raw/mosar --zenodo-dir data/raw/zenodo
```

Outputs:

- `models/goat_motion.joblib` — classifier, feature contract and gait baseline.
- `reports/metrics.json` — strict animal/source-grouped 70% training, 20% final
  testing and 10% validation results. Validation selects between Random Forest,
  regularized Random Forest and Extra Trees; final test animals remain untouched
  until selection is complete.

## Measured result and overfitting audit

The strict primary horn-profile test accuracy is **94.12%**, but this imbalanced headline
must not be interpreted as every class exceeding 90%. On held-out test goats the
per-class F1 scores are Grazing 99.03%, Resting 85.20%, Rumination 89.63% and
Walking 44.94%. Walking recall is 83.33%, but precision is only 30.77% because the
class has only 24 windows in the untouched test animals.

Five-fold group cross-validation, in which every fold holds out complete goats,
gives mean training accuracy 92.74% and mean unseen-goat accuracy 90.91%: a 1.84
percentage-point generalization gap. Mean class F1 is Grazing 98.23%, Resting
88.98%, Rumination 83.68% and Walking 44.09%. Grazing remains stable across unseen
goats. No finite dataset can prove zero overfitting, so training rejects candidates
whose train-to-unseen-validation accuracy gap exceeds three percentage points.
The results are saved in `reports/cross_validation.json`.

The improvement comes from 75 time/frequency features, including orientation-
invariant acceleration-magnitude spectral entropy, spectral centroid, frequency-band
energy and autocorrelation. These were selected because independent goat-wise research
found magnitude/frequency transformations more robust to sensor rotation.

## Predict

```powershell
python predict.py sensor_capture.csv --model models/goat_motion.joblib
```

The default is `--profile horn`. An ear-tag experiment can be run explicitly with
`--profile ear`, but its metrics in `reports/metrics.json` should be reviewed first.

Run the repeated unseen-goat overfitting audit with `python audit_cv.py`. This is
slower than the fixed 70/20/10 evaluation but reveals whether one favorable animal
split is inflating a class result.

For a 100 Hz BMI270 stream in m/s², conversion and resampling are explicit:

```powershell
python predict.py sensor_capture.csv --sample-rate 100 --acc-unit m/s2
```

The JSON output gives behavior probabilities and, on Walking windows, Normal gait
or Mobility anomaly. A production alert should require repeated anomalous windows
(for example 3 of 5), compare each goat with its own baseline, and combine piezo
hoof-strike features with daily trends.

## Piezo frontend test

Start the local dashboard and open `http://127.0.0.1:8080` in Chrome or Edge:

```powershell
python serve_frontend.py
```

The dashboard accepts newline-delimited serial values or JSON such as
`{"piezo": 245}` at 115200 baud. For a 12-bit ADC the initial UI defaults are:
baseline 0, light touch delta 30, press delta 120 and abnormal-force delta 700.
Use a resistor/protection circuit appropriate for the piezo because an unloaded
piezo can generate voltage spikes outside an MCU ADC's safe range. A large or clipped
impact becomes `abnormal_force`; steady pressure may decay because a piezo is a dynamic
force sensor. Piezo status is deliberately separate from Rumination.

## Refinement needed for the final device

For credible field accuracy, record each target goat with the final BMI270 + piezo
hardware and exact mounting position. Video-label at least several sessions per
animal, including different terrain and times of day. For the mobility model,
obtain veterinarian-scored sound and lame/foot-rot cases; split evaluation by goat
and farm. Accuracy from the horn-mounted public dataset is a pretraining benchmark,
not the expected accuracy of a differently mounted production sensor.
