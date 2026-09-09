"""
FastAPI serving layer for SkinSense.

Exposes:
    GET  /health   -- liveness check
    POST /predict  -- multipart image upload -> gated classification + Grad-CAM

The fine-tuned focal ConvNeXt model is loaded once, at process startup, via
`inference.get_ensemble()` -- not on every request.
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app import inference

app = FastAPI(
    title="SkinSense Inference API",
    description=(
        "Returns a triage classification for a dog skin-condition photo -- "
        "not a medical diagnosis. See app/README.md."
    ),
)

# Permissive CORS so any frontend origin (e.g. a separately-hosted React app)
# can call this API during development/demo.
# NOTE: this is fine for a dev/demo tool but should be locked down (specific
# allowed origins, not "*") before any real production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_models_on_startup() -> None:
    """
    Loads the checkpoint once when the server starts, so the (potentially
    slow) model construction and weight loading doesn't happen on the first
    request or repeatedly. Raises (and prevents the server from serving
    traffic) if the checkpoints are missing/invalid -- fail fast and loud
    rather than 500ing on every /predict call.
    """
    inference.get_ensemble()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Accepts a multipart/form-data upload with field name `file` containing an
    image, and returns the gated triage classification:

        {
            "predicted_class": str,
            "confidence": float,
            "per_class_probabilities": {class_name: float, ...},
            "low_confidence": bool,
            "quality_gate": object,
            "gradcam": object | null
        }
    """
    raw_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # force decode now, not lazily, so bad files fail here
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file could not be read as an image. Please upload a "
                "valid JPEG/PNG/etc. photo."
            ),
        )

    return inference.predict(image)
