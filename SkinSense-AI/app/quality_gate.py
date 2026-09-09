"""
Pre-inference quality gate for SkinSense (PROJECT.md item 9).

Runs three cheap checks on an uploaded image BEFORE it reaches the disease
classifier, so obviously-invalid inputs get a clear rejection reason instead
of a confident-looking disease label:

  1. Blur detection      -- variance of a Laplacian-filtered grayscale image.
                             Low variance = few sharp edges = likely blurry.
  2. Framing/size sanity  -- too small, absurd aspect ratio, or a flat/blank
                             (near-zero pixel variance) image.
  3. Out-of-distribution  -- how far the image's embedding (the classifier's
                             own penultimate-layer feature vector) sits from
                             every training class's centroid. A photo of
                             something that isn't dog skin at all (a cat, a
                             receipt, a screenshot) tends to land far from
                             all five centroids at once, which a max-softmax
                             score alone can miss (a confidently-wrong softmax
                             is exactly the failure mode this is for).

This module has no FastAPI/web-framework dependency -- app/api.py or
app/inference.py call `run_quality_gate()` before the main classifier.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter

LAPLACIAN_KERNEL = (0, 1, 0, 1, -4, 1, 0, 1, 0)

# Tuned against a constructed good/bad test set -- see models/quality_gate_eval.json
# for the false-positive/false-negative rates these thresholds produce.
BLUR_VARIANCE_MIN = float(os.environ.get("SKINSENSE_BLUR_MIN", "80"))
MIN_DIM_PX = int(os.environ.get("SKINSENSE_MIN_DIM", "64"))
MAX_ASPECT_RATIO = float(os.environ.get("SKINSENSE_MAX_ASPECT", "4.0"))
FLAT_IMAGE_STD_MIN = float(os.environ.get("SKINSENSE_FLAT_STD_MIN", "8.0"))
OOD_CENTROID_ZSCORE_MAX = float(os.environ.get("SKINSENSE_OOD_ZSCORE_MAX", "3.0"))


@dataclass
class QualityGateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blur_variance: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    pixel_std: Optional[float] = None
    ood_zscore: Optional[float] = None
    nearest_class: Optional[str] = None

    def to_dict(self):
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "diagnostics": {
                "blur_variance": self.blur_variance,
                "width": self.width,
                "height": self.height,
                "pixel_std": self.pixel_std,
                "ood_zscore": self.ood_zscore,
                "nearest_class": self.nearest_class,
            },
        }


def blur_variance(image: Image.Image) -> float:
    """Variance of a Laplacian-filtered grayscale image. Low = blurry."""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.Kernel((3, 3), LAPLACIAN_KERNEL, scale=1))
    arr = np.asarray(edges, dtype=np.float64)
    return float(arr.var())


def framing_checks(image: Image.Image):
    """Returns (ok: bool, reasons: list[str], width, height, pixel_std)."""
    reasons = []
    w, h = image.size
    if w < MIN_DIM_PX or h < MIN_DIM_PX:
        reasons.append(f"image too small ({w}x{h}px, minimum {MIN_DIM_PX}px on each side)")
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
        reasons.append(f"unusable aspect ratio ({w}x{h}, {aspect:.1f}:1)")
    gray_arr = np.asarray(image.convert("L"), dtype=np.float64)
    pixel_std = float(gray_arr.std())
    if pixel_std < FLAT_IMAGE_STD_MIN:
        reasons.append(f"image appears blank/flat (pixel std={pixel_std:.1f})")
    return (len(reasons) == 0), reasons, w, h, pixel_std


@torch.no_grad()
def embed(model, transform, image: Image.Image, device):
    """Penultimate-layer embedding (pre-classifier-head pooled features)."""
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        features = model.forward_features(x)
        pooled = model.forward_head(features, pre_logits=True)
    return pooled.float().cpu().numpy()[0]


class OODDetector:
    """Nearest-centroid out-of-distribution check using training-set embeddings.

    Loads per-class centroids + within-class distance stats produced by
    scripts/compute_centroids.py. A query image is flagged OOD if its distance
    to the NEAREST class centroid is more standard deviations away than
    OOD_CENTROID_ZSCORE_MAX, using that centroid's own within-class spread.
    """

    def __init__(self, centroids_path: str):
        if not os.path.isfile(centroids_path):
            raise FileNotFoundError(
                f"OOD centroids file not found at '{centroids_path}'. Run "
                "scripts/compute_centroids.py against the shipped checkpoint first."
            )
        data = np.load(centroids_path, allow_pickle=True)
        self.classes = list(data["classes"])
        self.centroids = data["centroids"]          # (n_classes, dim)
        self.within_class_mean = data["within_class_mean"]
        self.within_class_std = data["within_class_std"]

    def score(self, embedding: np.ndarray):
        dists = np.linalg.norm(self.centroids - embedding[None, :], axis=1)
        nearest_idx = int(dists.argmin())
        z = (dists[nearest_idx] - self.within_class_mean[nearest_idx]) / max(
            self.within_class_std[nearest_idx], 1e-6
        )
        return float(z), self.classes[nearest_idx]


def run_quality_gate(image: Image.Image, model=None, transform=None, device=None,
                      ood_detector: Optional[OODDetector] = None) -> QualityGateResult:
    """
    Runs blur + framing checks always; runs the OOD check only if a model,
    transform, device, and ood_detector are all provided (keeps this testable
    without a loaded model for the blur/framing-only path).
    """
    reasons = []
    warnings = []
    blur_var = blur_variance(image)
    if blur_var < BLUR_VARIANCE_MIN:
        reasons.append(f"image appears too blurry (edge variance={blur_var:.1f}, minimum {BLUR_VARIANCE_MIN})")

    frame_ok, frame_reasons, w, h, pixel_std = framing_checks(image)
    reasons.extend(frame_reasons)

    ood_z = None
    nearest_class = None
    if model is not None and transform is not None and ood_detector is not None:
        emb = embed(model, transform, image, device)
        ood_z, nearest_class = ood_detector.score(emb)
        if ood_z > OOD_CENTROID_ZSCORE_MAX:
            # This detector caught 0/10 synthetic wrong-subject images in the
            # constructed evaluation. Keep the signal visible to callers, but
            # do not let an unvalidated semantic-OOD heuristic reject uploads.
            warnings.append(
                f"image is unusually distant from the trained class embeddings "
                f"(z={ood_z:.2f} from nearest class '{nearest_class}', threshold {OOD_CENTROID_ZSCORE_MAX}) "
                "-- experimental OOD warning only; not a reliable wrong-subject detector"
            )

    return QualityGateResult(
        passed=(len(reasons) == 0),
        reasons=reasons,
        warnings=warnings,
        blur_variance=blur_var,
        width=w, height=h, pixel_std=pixel_std,
        ood_zscore=ood_z, nearest_class=nearest_class,
    )
