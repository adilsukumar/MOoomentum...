# Dog Health Vitals Dataset

Source: https://doi.org/10.5281/zenodo.8020390

This directory is reserved for the public Invoxia Dog Health Vitals Dataset.
The downloaded source files are distributed under CC BY 4.0; cite:

> Jarkoff, H., Lorre, G., & Humbert, E. (2023). Assessing the Accuracy of a Smart Collar for Dogs: Predictive Performance for Heart and Breathing Rates on a Large Scale Dataset. bioRxiv. https://doi.org/10.1101/2023.06.09.544347

Expected source files:

- `dataset.csv` — session metadata plus HR/RR segments, ECG beat times, and poor-quality ECG intervals.
- `ecg_data.zip` — ECG waveform files referenced by `dataset.csv`.
- `SOURCE_README.md` — upstream data dictionary and license.

Verified local contents:

- 1,123 session rows from 40 dogs (93.09 total recording hours).
- 1,102 referenced ECG WAV files extracted under `ecg_data/`.
- 21 session rows have no `ecg_path`; every non-empty ECG path resolves locally.
- 361,672 ECG pulse timestamps.
- Download MD5 values match Zenodo: `dataset.csv` = `fcde93215fac9273ab7a5ffc10357de2`; `ecg_data.zip` = `ce1a8bffda45c68250a061f12260f442`.

This is a validation/test dataset for sleeping dogs. It does **not** contain VitalSense Piezo or BMI270 signals and should not be presented as a complete training dataset for the proposed hardware.
