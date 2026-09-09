# Dog Health Vitals Dataset

## Description
The Dog Health Vitals Dataset is a collection of recordings and vital statistics related to the health of dogs. It includes data captured during various recording sessions, providing insights into the physiological characteristics of the dogs.

The dataset is presented in CSV format, with each line representing a recording session. The CSV file contains the following columns:

- `_id`: Unique identifier for each recording session.
- `ecg_path`: Relative path to the corresponding ECG file (in WAV format).
- `duration`: Duration of the recording session (in seconds).
- `pet_id`: Unique identifier for each dog.
- `breeds`: Main breed of the dog.
- `weight`: Weight of the dog at the time of measurement (in kg).
- `age`: Age of the dog at the time of measurement (in years).
- `segments_br`: Array of dictionaries, each dict representing a breathing rate on a specific time segment of the signal. Each dict has three keys : `deb` representing the beginning of the segment (in seconds from the beginning of the signal), `fin` representing the end of the segment (in seconds from the beginning of the signal), and `value` representing the value of the breathing rate on this segment.
- `segments_hr`: Array of dictionaries, each dict representing a heart rate on a specific time segment of the signal. Each dict has three keys : `deb` representing the beginning of the segment (in seconds from the beginning of the signal), `fin` representing the end of the segment (in seconds from the beginning of the signal), and `value` representing the value of the heart rate on this segment.
- `ecg_pulses`: Array of floats, each representing the timestamp (in seconds from the beginning of the signal) of an identified heart pulse on the ECG signal.
- `bad_ecg`: Array of tuples, representing time segments of poor ECG signal quality. Each tuple has two elements, the first being the beginning of the segment (in seconds from the beginning of the signal), and the second the end of this segment (in seconds from the beginning of the signal).
  

## Data Files
The dataset archive includes the following files:

- `dataset.csv`: The main dataset file in CSV format.
- ECG waveform files, in `ecg_data` directory: The corresponding ECG waveform files referenced in the dataset. The paths to these files are provided in the `ecg_path` column of the CSV file.

## License
This dataset is made available under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. By using this dataset, you agree to the terms and conditions specified in the license.

## Citation
If you use this dataset in your research or any other publication, please cite it as:

`Jarkoff, H., Lorre, G., & Humbert, E. (2023). Assessing the Accuracy of a Smart Collar for Dogs: Predictive Performance for Heart and Breathing Rates on a Large Scale Dataset. Preprint available on bioRxiv.`


## Contact Information
For any questions or inquiries regarding the dataset, please contact:

Invoxia Research
<br>
research@invoxia.com
<br>
Invoxia, 8 Esp. de la Manufacture, 92130 Issy-les-Moulineaux, France
