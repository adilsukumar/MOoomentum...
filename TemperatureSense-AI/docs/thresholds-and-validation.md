# Temperature thresholds, evidence, and validation

## 1. What the outputs mean

The bands below are conservative **product alert bands**, not universal canine
physiologic limits. An environmental `NORMAL` result means only that no risk
rule fired for the supplied inputs. It must not be displayed as “safe.” Sun,
ground temperature, wind, enclosure, hydration, health, acclimation, and recent
activity can dominate the SHT40 reading.

### External SHT40 base bands

| Local air temperature | Base output | Upper/lower use |
|---:|---|---|
| below -5 °C | extreme cold | critical base alert |
| -5 to -0.1 °C | very cold | high base alert |
| 0 to 9.9 °C | cold | caution base alert |
| 10 to 24.9 °C | moderate | no base alert; context can still raise risk |
| 25 to 29.9 °C | warm | caution base alert |
| 30 to 34.9 °C | hot | high base alert |
| 35 °C or above | extreme heat | critical base alert |

Risk is raised for high humidity, exertion, direct sun, poor airflow, enclosed
vehicles, brachycephaly, obesity, very large size, vulnerable age, heavy coat,
and respiratory/cardiac disease. These boundaries are editable engineering
defaults that require field validation; no formal universal canine working-heat
guideline exists. One VetCompass study found a median ambient temperature of
only 16.9 °C on heat-related illness event days because exertion and individual
risk matter. A separate military-working-dog review reported events down to a
WBGT of 20.4 °C. The SHT40 cannot measure WBGT because it lacks globe/radiant
temperature and wind speed.

### NTC surface-contact trend bands

| Difference from the dog's healthy sleep median | Output |
|---:|---|
| below -1.5 °C | well below baseline |
| -1.5 to -0.8 °C | below baseline |
| -0.8 to +0.8 °C | personal baseline range |
| +0.8 to +1.5 °C | above baseline |
| above +1.5 °C | well above baseline |

These deltas are initial surface-anomaly settings, not clinical fever limits.
The sample window requires at least five readings and rejects a contact swing
over 1.0 °C. A robust personal baseline is the median of one stable-window
median from each of at least seven known-healthy sleeping nights. The NTC
surface value is never converted to internal/core temperature.

### Values calculated from the NTC

| Calculation | Purpose | User-facing output |
|---|---|---|
| ADC count → resistance | Apply the exact divider and thermistor constants | not displayed |
| resistance → surface °C | Physical NTC surface-contact measurement | stored for processing |
| median of at least five samples | Reject short noise spikes | range only |
| maximum minus minimum sample | Detect unstable collar contact | stable/unstable |
| median minus personal healthy baseline | Individual anomaly detection | below/baseline/above range |
| delta band + contact quality | Select alert severity and action | normal/caution/high/invalid |

No internal temperature, fever grade, disease, or rabies probability is
calculated from the NTC.

## 2. Breed handling

There is no scientifically defensible “temperature range for every breed.” The
included catalog preserves the 2020 UK VetCompass population odds ratios for
breeds present in that study. It does not invent values for unstudied breeds and
does not convert odds ratios into degrees Celsius.

The significantly higher heat-illness odds compared with Labrador Retrievers
were reported for Chow Chow, Bulldog, French Bulldog, Dogue de Bordeaux,
Greyhound, Cavalier King Charles Spaniel, Pug, English Springer Spaniel, and
Golden Retriever. Population association is not destiny: a mixed-breed dog with
brachycephaly, obesity, airway disease, or poor acclimation may be at greater
risk than its breed label suggests. Ask for explicit traits during onboarding.

The Husky example is useful here: the study did not show significantly greater
overall UK heat-illness odds for Siberian Huskies, so assigning a fabricated
“Husky maximum” would be misleading. The engine still accounts for a heavy
double coat and activity and will escalate warm/humid exposures.

## 3. Rabies rule

**Never output `rabies detected` from surface temperature.** Rabies is an acute
neurologic disease and official decisions depend on exposure history,
progressive clinical signs, public-health assessment, and approved laboratory
testing.

The implemented logic is:

1. Surface-temperature anomaly alone → `NO_RABIES_INFERENCE`.
2. Surface anomaly plus abrupt behavior/oral signs → urgent nonspecific illness
   alert and veterinary review, not a rabies result.
3. Known/possible rabies exposure, or a progressive neurologic/sign cluster →
   avoid saliva/bites, isolate safely, and immediately contact a veterinarian
   and local public-health/animal-control authority. Do not wait for a
   temperature change.
4. Any person bitten or scratched should wash the wound and obtain urgent human
   medical advice under the local rabies protocol.

Do not train an AI model with “rabies” labels inferred from temperature. That
would encode false labels and create a dangerous false-negative system.

## 4. Hardware and calibration requirements

### SHT40

- Put it on the outside of the enclosure behind a hydrophobic, vapor-permeable
  vent, away from the dog's body, NTC, battery, regulator, MCU, and radio.
- Characterize self-heating and enclosure lag in a chamber. Sensirion specifies
  typical temperature accuracy of ±0.2 °C and typical RH accuracy of ±1.8 %RH,
  but the finished enclosure is a different measurement system.
- Direct sun makes the package temperature differ from air temperature. Supply
  a sun/radiant-heat context flag or add suitable radiative shielding.
- Never treat a collar reading as the temperature throughout a parked vehicle.

### 10 kΩ NTC

- Use the exact part's resistance tolerance, B-value/Steinhart–Hart constants,
  dissipation constant, and response time. “10 kΩ” alone is insufficient.
- Calibrate each analog front end in a stirred temperature bath across the
  expected range with a traceable reference thermometer.
- Keep excitation low or duty-cycled to prevent thermistor self-heating.
- Use a smooth, biocompatible, cleanable contact surface with no sharp pressure
  point. Collar fit must be repeatable and must not restrict breathing.
- A contact sensor over fur measures the local contact/fur microclimate. On
  hairless skin it measures local skin-surface contact temperature. Neither is
  treated as internal temperature.

## 5. Required validation before health claims

1. **Bench validation:** ADC, resistor tolerance, NTC conversion, open/short
   detection, CRC failures, condensation, battery voltage, and temperature lag.
2. **Per-dog baseline:** at least seven healthy nights, same sensor position and
   collar fit, after five minutes of BMI270-confirmed sleep. Freeze a robust
   median; do not let an illness automatically drift the baseline upward.
3. **Surface-trend field study:** compare repeatability across breeds, coats,
   sizes, collar fits, environments, and normal/abnormal health periods. Measure
   false-alert and missed-alert rates for the surface anomaly bands. Do not fit
   or advertise a conversion to internal temperature.
4. **Environmental field study:** test shade, sun, wind, high humidity, wet fur,
   hot pavement, vehicles, and exercise. Add globe/ground sensing if those claims
   matter.
5. **Human factors:** ensure “range” messages do not delay cooling, veterinary
   treatment, rabies exposure reporting, or bite care.

## 6. Sources

- Hall EJ, et al. canine heat-illness incidence and risk factors (2020):
  https://doi.org/10.1038/s41598-020-66015-8
- Hall EJ, et al. severe/fatal heat-illness risk and ambient conditions (2022):
  https://doi.org/10.3390/vetsci9050231
- Rizzo M, et al. surface versus rectal temperature in dogs (2020):
  https://doi.org/10.1016/j.vas.2020.100120
- CDC information for veterinarians—rabies signs and exposure management:
  https://www.cdc.gov/rabies/hcp/veterinarians/index.html
- Sensirion SHT40 specifications and datasheet:
  https://sensirion.com/products/catalog/SHT40
