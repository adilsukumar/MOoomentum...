# SkinSense inference API

This directory contains the serving layer that a frontend app calls to get a
prediction for a dog skin-condition photo. It wraps the final focal-loss
ConvNeXt-Tiny model trained on 4 classes: `Allergic_dermatitis`,
`Demodicosis`, `Fungal_infection`, `Healthy`. `Bacterial_dermatosis` was
removed from the output head after a data-quality and reliability audit.

## What this is (and isn't)

**This endpoint returns a triage classification, not a medical diagnosis.**
It gives a dog owner a fast first signal to help decide how urgently to see a
vet -- it does not replace veterinary examination or diagnosis. See
`PROJECT.md` section 3 ("MVP scope") and decisions D4/D5 at the repo root for
the full product framing.

When the response has `"low_confidence": true`, the frontend should show the
user something like **"unclear -- please consult a vet"** rather than the
`predicted_class` value presented as a confident-sounding disease name. The
model is not confident enough in that case for the class label to be
trustworthy on its own.

## Starting the server

From the repo root (`D:\22_SkinSense_ML`), run:

```powershell
& "D:\22_SkinSense_ML\.venv\Scripts\python.exe" -m uvicorn app.api:app --reload --port 8000
```

The server loads both model checkpoints once at startup (not per-request).
If either checkpoint is missing, startup will fail with a clear
`FileNotFoundError` naming the expected path -- this is expected until both
training runs have finished and written:

- `D:\22_SkinSense_ML\models\effnet\best.pt`
- `D:\22_SkinSense_ML\models\convnext\best.pt`

Checkpoint paths can be overridden without editing code via the environment
variable `SKINSENSE_CONVNEXT_CKPT`. The
low-confidence threshold can likewise be overridden via
`SKINSENSE_CONF_THRESHOLD` (default `0.60` -- see "Confidence threshold"
below).

## Endpoints

### `GET /health`

Liveness check.

**Response:**
```json
{ "status": "ok" }
```

### `POST /predict`

Accepts a single image as `multipart/form-data` with field name **`file`**.
Returns the classification, quality-gate diagnostics, and Grad-CAM as JSON.

**Response shape:**
```json
{
  "prediction_status": "classified",
  "predicted_class": "Fungal_infection",
  "confidence": 0.812,
  "per_class_probabilities": {
    "Allergic_dermatitis": 0.041,
    "Demodicosis": 0.038,
    "Fungal_infection": 0.812,
    "Healthy": 0.057
  },
  "low_confidence": false,
  "quality_gate": {
    "passed": true,
    "reasons": [],
    "diagnostics": {
      "blur_variance": 174.7,
      "width": 640,
      "height": 640,
      "pixel_std": 64.6,
      "ood_zscore": -0.93,
      "nearest_class": "Fungal_infection"
    }
  },
  "gradcam": {
    "class": "Fungal_infection",
    "image_data_uri": "data:image/png;base64,...",
    "method": "Grad-CAM",
    "disclaimer": "Shows model attention, not a lesion boundary or clinical explanation."
  }
}
```

If the quality gate rejects an image, `prediction_status` is
`"rejected_quality_gate"`, disease/probability fields are empty, and `gradcam`
is `null`. The current gate reliably catches tiny, blank, and extreme-aspect
uploads, but its tested wrong-subject/OOD detector is not reliable; see
`models/quality_gate_full_eval.json` and PROJECT.md D15.

If the uploaded file isn't a readable image, the endpoint responds with
`HTTP 400` and a JSON body describing the problem, e.g.:
```json
{ "detail": "Uploaded file could not be read as an image. Please upload a valid JPEG/PNG/etc. photo." }
```

#### curl example

```bash
curl -F "file=@dog_photo.jpg" http://localhost:8000/predict
```

#### JavaScript (fetch) example

```javascript
async function getPrediction(file) {
  const formData = new FormData();
  formData.append("file", file); // `file` is a File/Blob, e.g. from <input type="file">

  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Prediction request failed");
  }

  const result = await response.json();

  if (result.low_confidence) {
    // Show "unclear -- please consult a vet" instead of result.predicted_class.
  }

  return result;
}
```

## Confidence threshold

`low_confidence` is `true` whenever model confidence is below `0.60`. On the
held-out focal-model test set, 91.9% of images clear this threshold and those
accepted predictions are 95.7% accurate. This is dataset-specific evidence,
not a guarantee for arbitrary phone photos.

## Known limitations (fine for local demo, not for production)

- **CORS is wide open** (`allow_origins=["*"]`) so any frontend can call this
  API during development. Lock this down to specific origins before any real
  deployment.
- **No rate limiting.**
- **No authentication.**
- **No file-size or file-type validation** beyond "can PIL decode this as an
  image." A very large upload will be read fully into memory.

None of the above is acceptable for a public deployment -- this API is
currently scoped as a local/dev-demo inference service only.
