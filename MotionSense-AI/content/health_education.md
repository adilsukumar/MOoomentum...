# Health Education Panel — Content & Framing Rules

> This is app copy, not a model. Nothing here is generated from sensor data. It exists so an "unusual pattern detected" flag from the anomaly detector (see [scripts/anomaly_detection.py](../scripts/anomaly_detection.py)) can link to *general* information without the app ever claiming to diagnose the cause.

## Hard rule for engineering and copywriting

The model NEVER outputs a disease name. The model only ever outputs one of three things:
1. A behavior label (Active / Resting / Other, or a fine-grained behavior).
2. An "unusual movement pattern" flag with a plain mechanical reason (e.g., "gait less regular than usual during a locomotion window," "shaking-type motion much more frequent than this dog's usual rate").
3. A confidence/frequency number attached to (1) or (2).

The educational panel is static content, shown only *after* an anomaly flag fires, and is never personalized beyond "here's general information about conditions that can involve gait or repetitive-motion changes — this app has not diagnosed your dog with anything." No condition below should ever be presented as "your dog might have X."

## What every anomaly flag screen must say, verbatim intent

> "MotionSense noticed a movement pattern that's unusual for [dog name]. This is not a diagnosis — sensors on a collar cannot identify a medical condition. If this continues, seems severe, or you're worried, please contact your veterinarian."

For a **rabies-relevant severity tier** specifically (see below), the copy must escalate to urgent/emergency framing rather than a routine "consider a vet visit" — this is the one condition in this list where a delay has public-health consequences, not just animal-welfare ones.

## General educational content (shown as reference material, not a result)

| Category | Behavioral/motor signs sometimes associated with it (general veterinary knowledge) | Anomaly flag it *loosely* relates to |
|---|---|---|
| Joint pain / arthritis / injury | Limping, gait asymmetry, reluctance to jump or climb stairs, stiffness after rest | Gait-regularity flag |
| Anxiety / compulsive disorder | Repetitive pacing, excessive licking or self-directed motion | Elevated shaking/repetitive-motion rate flag |
| Skin conditions, allergies, parasites (fleas, mites) | Frequent scratching or biting at skin, excessive shaking | Elevated shaking/repetitive-motion rate flag |
| Gastrointestinal upset | Restlessness, pacing, lip licking | General anomaly (IsolationForest) flag |
| General illness, pain, or heat stress | Reduced activity, lethargy, reluctance to move | Drop in "Active" time flag |
| Neurological conditions (seizure disorders, vestibular disease, etc.) | Disorientation, loss of coordination (ataxia), tremors | Gait-regularity flag, general anomaly flag |
| **Rabies** *(emergency tier — see below)* | Behavioral change (unusual aggression **or** unusual tameness/affection), disorientation, incoordination/ataxia, hind-limb weakness progressing to paralysis, excessive drooling, difficulty swallowing | Gait-regularity flag + general anomaly flag occurring together, or any flag alongside owner-observed behavioral change |

## Rabies-specific framing (read before writing any related UI copy)

- Rabies is nearly always fatal once clinical signs appear, and it is a **public health emergency** — a person bitten by a potentially rabid animal needs timely medical evaluation for post-exposure prophylaxis, independent of anything this app says.
- This app **cannot** and does not test for rabies. There is no rabies-labeled training data anywhere in this project, and a wearable accelerometer cannot detect a virus.
- The only responsible role for the app here is: (a) general education on what rabies signs can look like, sourced from public-health/veterinary guidance, and (b) an unambiguous instruction that any suspected case — bite exposure, unexplained behavioral change plus incoordination, unprovoked aggression, or known exposure to a rabid/wild animal — should go **immediately** to a veterinarian or local animal control/public health authority, not be monitored via the app.
- Suggested copy for this tier: *"Some of the patterns above can, in rare cases, be associated with serious conditions including rabies. This app cannot detect rabies or any other disease. If your dog has had contact with a wild animal, has an unexplained bite wound, or is showing sudden behavioral change together with movement problems, contact a veterinarian or animal control immediately — do not wait to see if the app flags anything else."*

## Open items

- [ ] Have this copy reviewed by an actual veterinary professional before shipping — this document was drafted from general public knowledge, not clinical guidance, and should not be treated as medical/veterinary authorship.
- [ ] Decide whether the rabies-emergency copy should be shown proactively (e.g., in an FAQ/help section) rather than only after a flag, since the message ("seek immediate care if X") is arguably more useful before a scary flag ever fires.
