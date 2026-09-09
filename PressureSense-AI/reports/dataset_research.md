# Goat MotionSense Dataset and Generalization Assessment

## Result

The production horn-profile classifier now exceeds the requested 90% overall target
under both evaluation protocols while retaining a strict animal boundary:

| Evaluation | Accuracy | Training accuracy | Generalization gap |
|---|---:|---:|---:|
| Untouched 20% animal-grouped test | 94.12% | — | — |
| Five-fold GroupKFold by goat | 90.91% | 92.74% | 1.84 points |
| Fixed unseen-animal validation | 91.98% | 92.64% | 0.66 points |

The split remains 70% training, 20% final testing and 10% validation. Complete goats,
not randomly mixed sensor windows, are assigned to each partition. Candidate models
with a training-to-validation accuracy gap above three percentage points are rejected.
This controls measured overfitting; no finite observational dataset can prove that
overfitting is exactly zero for every future farm, breed or sensor mounting.

## Data inventory and decision

CabriTrack is the primary training source because it contains more than 144 annotated
hours from 59 goats, uses 25 Hz horn-mounted triaxial accelerometers, and directly
labels grazing, displacement, rumination/chewing and resting.^1 It is the only public
source found that closely matches all four target behaviours and has enough independent
goats for meaningful held-animal evaluation.

The complete supplied UPV Zenodo archive was downloaded and integrated as a separate
neck-profile expert. It supplies eating, walking/moving, inactivity/lying and rumination
labels, but eating is not equivalent to pasture grazing and the mounting differs from
CabriTrack.^2 MoSAR recordings from eight goats were also downloaded and retained as
an experimental ear-profile expert.^3 Keeping these sources separate prevents a model
from learning dataset or mounting identity instead of behaviour.

The five-goat Kamminga collar dataset was investigated specifically for additional
walking examples.^4 One 74 MB raw recording was downloaded for schema inspection.
The complete position-A collection was not merged into the horn model: it is a 100 Hz
neck/collar domain, lacks rumination, and an independent re-evaluation reports only
79.4% best goat-wise balanced accuracy despite mixed-window results above 94%.^5 This
is direct evidence that merging it and randomly splitting windows could inflate the
headline while weakening new-goat validity.

The Dryad pygmy-goat/caprid dataset was also evaluated. Its own publication reports
greater than 98% accuracy with ordinary splitting but only 56.1% ± 11% when complete
individuals are held out.^6 It is processed, differently mounted surrogate data and
therefore was not used to manufacture a higher production score.

## Model improvement

The raw accelerometer feature set was expanded from 49 to 75 features. New features
include acceleration-magnitude spectral entropy, spectral centroid, four frequency-band
energy ratios and one- and two-second autocorrelation, with equivalent spectral features
for each axis. The rationale is supported by the independent goat-wise Kamminga analysis,
where magnitude and frequency-domain transformations reduced orientation dependence.^5

A narrowly bounded grouped-CV search compared adjacent tree depths and minimum leaf
sizes. The selected Random Forest has maximum depth 5 and at least 60 training windows
per leaf. It achieved the best accepted combination: more than 90% mean unseen-goat
accuracy and less than a three-point mean generalization gap. The final 20% test animals
were not used for feature choice, threshold choice or model selection.

## Remaining limitation

Overall accuracy is above 90%, but every class is not. Five-fold mean F1 is 98.23% for
Grazing, 88.98% for Resting, 83.68% for Rumination and 44.09% for Walking. On the fixed
test split, Walking recall is 83.33%, but precision is 30.77%; only 24 walking windows
occur in those test goats. The CabriTrack definition also combines walking, running and
posture transitions as “Displacement.”^1 More public data cannot correct that target-label
ambiguity. A reliable greater-than-90% Walking F1 requires new video-labelled BMI270
data collected with the final mounting, across independent goats and farms.

Mobility anomaly remains an unsupervised screen, not a validated foot-rot classifier.
No compatible public dataset with raw goat BMI270/piezo signals and veterinarian-scored
normal versus lame/foot-rot labels was found.

## Sources

1. Alvarenga et al. “[CabriTrack: Accelerometer data for automated behavioural monitoring of grazing Creole goats](https://pmc.ncbi.nlm.nih.gov/articles/PMC11953975/).” 2025.
2. Universitat Politècnica de València. “[Goat accelerometer data](https://doi.org/10.5281/zenodo.17853479).” Zenodo.
3. Duvaux-Ponter and Taghipoor. “[Accelerometers and annotated video recordings of behaviours: a dataset for training behaviour classification models in goats](https://doi.org/10.57745/LGZBM1).” Recherche Data Gouv, 2024.
4. Kamminga. “[Multi Sensor-Orientation Movement Data of Goats](https://doi.org/10.17026/dans-xhn-bsfb).” DANS, 2018.
5. Hollevoet et al. “[Goats on the Move: Evaluating Machine Learning Models for Goat Activity Analysis Using Accelerometer Data](https://doi.org/10.3390/ani14131977).” Animals, 2024.
6. Dickinson et al. “[Limitations of using surrogates for behaviour classification of accelerometer data](https://doi.org/10.1186/s40462-021-00265-7).” Movement Ecology, 2021.
