"""
Inference layer for SkinSense.

Serves the fine-tuned ConvNeXt-Tiny focal-loss classifier (built via `timm`)
on the final 4-class taxonomy. Bacterial_dermatosis is intentionally absent
from the classification head (PROJECT.md D8/D9).

Why not the EfficientNetV2 + ConvNeXt ensemble originally planned: both
backbones were fine-tuned and measured on the same held-out test set
(models/ensemble_test_metrics.json). ConvNeXt-Tiny alone scored 80.6%
accuracy / 0.71 macro-F1 versus EfficientNetV2's 66.6% / 0.65. A full sweep
of blend weights (models/ensemble_test_metrics.json's sibling analysis)
showed every blend underperforms ConvNeXt-Tiny alone -- EfficientNetV2 was
never strong enough to add value, only to drag the average down. See
PROJECT.md decision D6. The EfficientNetV2 checkpoint is left on disk
(models/effnet/) in case a future retrain makes it worth revisiting.

This module does not run any web framework code itself -- see app/api.py for
the FastAPI wrapper that calls `predict()`.
"""

import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

from app.gradcam import gradcam_overlay_for_image
from app.quality_gate import OODDetector, run_quality_gate

# ---------------------------------------------------------------------------
# Configuration -- override via environment variable if the checkpoint lives
# somewhere other than the final focal checkpoint layout.
# ---------------------------------------------------------------------------
CONVNEXT_CKPT_PATH = os.environ.get(
    "SKINSENSE_CONVNEXT_CKPT",
    os.path.join("models", "convnext_4class_focal", "best.pt"),
)
CENTROIDS_PATH = os.environ.get(
    "SKINSENSE_CENTROIDS_PATH",
    os.path.join("models", "convnext_4class_focal", "quality_centroids.npz"),
)

# Below this confidence, the caller should treat the result as "unclear" and
# steer the user toward a vet rather than a specific class. Tuned empirically
# against the held-out focal-model test probabilities: at 0.60, 91.9% of test
# images clear the bar and are 95.7% accurate when they do; the remaining 8.1%
# are routed to the fallback. See PROJECT.md decision D5.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("SKINSENSE_CONF_THRESHOLD", "0.60"))

# Kept as an extension point for future per-class abstention rules. Bacterial
# dermatosis no longer needs a rule because it is not an output class at all.
UNRELIABLE_CLASSES = set()

# ImageNet normalization stats -- must match how the backbones were
# pretrained/fine-tuned.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _LoadedModel:
    """Holds one fine-tuned model plus everything needed to run it standalone."""

    def __init__(self, checkpoint_path: str):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"SkinSense model checkpoint not found at '{checkpoint_path}'. "
                "Training has likely not finished yet -- this file is expected to "
                "be produced by the fine-tuning script as a torch.save()'d dict "
                "with keys 'state_dict', 'arch', 'classes', and 'img_size'. "
                "Inference cannot run until it exists."
            )

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )

        for required_key in ("state_dict", "arch", "classes", "img_size"):
            if required_key not in checkpoint:
                raise KeyError(
                    f"Checkpoint '{checkpoint_path}' is missing required key "
                    f"'{required_key}'. Expected keys: state_dict, arch, classes, "
                    "img_size."
                )

        self.arch: str = checkpoint["arch"]
        self.classes: List[str] = list(checkpoint["classes"])
        self.img_size: int = int(checkpoint["img_size"])

        self.model = timm.create_model(
            self.arch, pretrained=False, num_classes=len(self.classes)
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(_DEVICE)

        # Resize slightly larger than the crop, matching the standard
        # ImageNet-style eval transform (crop ratio ~0.875, i.e. resize to
        # img_size / 0.875 ~= img_size * 1.14).
        resize_to = int(round(self.img_size * 1.14))
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_to),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def predict_probs(self, image: Image.Image) -> Dict[str, float]:
        """Runs a single PIL image through this model, returns class -> prob."""
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(_DEVICE)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        return dict(zip(self.classes, probs))


class Ensemble:
    """
    Loads the classifier once and exposes a single `predict()` method.

    Named `Ensemble` for API stability (app/api.py and any frontend code
    already refer to `get_ensemble()`/`predict()`), even though it currently
    wraps a single model -- see the module docstring for why the planned
    two-model ensemble was dropped in favor of ConvNeXt-Tiny alone.
    """

    def __init__(self, convnext_ckpt_path: str = CONVNEXT_CKPT_PATH):
        self.convnext = _LoadedModel(convnext_ckpt_path)
        self.classes: List[str] = sorted(self.convnext.classes)
        self.ood_detector = (
            OODDetector(CENTROIDS_PATH) if Path(CENTROIDS_PATH).is_file() else None
        )

    def predict(self, image: Image.Image) -> dict:
        """
        Runs `image` through the model and returns the result.

        Returns a dict:
            {
                "predicted_class": str,
                "confidence": float,
                "per_class_probabilities": {class_name: float, ...},
                "low_confidence": bool,
            }
        """
        gate = run_quality_gate(
            image,
            model=self.convnext.model,
            transform=self.convnext.transform,
            device=_DEVICE,
            ood_detector=self.ood_detector,
        )
        if not gate.passed:
            return {
                "prediction_status": "rejected_quality_gate",
                "predicted_class": None,
                "confidence": None,
                "per_class_probabilities": {},
                "low_confidence": True,
                "quality_gate": gate.to_dict(),
                "gradcam": None,
            }

        probs = self.convnext.predict_probs(image)

        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class]

        # Force the known-unreliable classes down the low-confidence path
        # regardless of raw softmax score (see UNRELIABLE_CLASSES above).
        low_confidence = (
            confidence < LOW_CONFIDENCE_THRESHOLD
            or predicted_class in UNRELIABLE_CLASSES
        )

        target_idx = self.convnext.classes.index(predicted_class)
        overlay, _ = gradcam_overlay_for_image(
            self.convnext.model,
            self.convnext.transform,
            image,
            _DEVICE,
            target_class_idx=target_idx,
        )

        return {
            "prediction_status": "classified",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "per_class_probabilities": probs,
            "low_confidence": low_confidence,
            "quality_gate": gate.to_dict(),
            "gradcam": {
                "class": predicted_class,
                "image_data_uri": overlay,
                "method": "Grad-CAM",
                "disclaimer": "Shows model attention, not a lesion boundary or clinical explanation.",
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton + convenience function.
#
# NOTE: instantiating `Ensemble()` here at import time would raise
# FileNotFoundError until both training runs finish producing checkpoints.
# We defer construction to `get_ensemble()` / `predict()` so this module can
# be imported (e.g. for testing, or by api.py before startup) without the
# checkpoints existing yet. app/api.py is responsible for calling
# `get_ensemble()` once at startup so the models are only loaded a single
# time, not per-request.
# ---------------------------------------------------------------------------
_ensemble: "Ensemble | None" = None


def get_ensemble() -> Ensemble:
    """Returns the process-wide Ensemble singleton, loading it on first call."""
    global _ensemble
    if _ensemble is None:
        _ensemble = Ensemble()
    return _ensemble


def predict(image: Image.Image) -> dict:
    """Convenience wrapper: predict() using the lazily-loaded singleton ensemble."""
    return get_ensemble().predict(image)
