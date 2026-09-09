"""
Grad-CAM saliency overlay for SkinSense (PROJECT.md item 10).

Works for any timm model exposing `forward_features()` (spatial feature map)
and `forward_head()` (pool + classify) -- ConvNeXt-family models included.
Computes the standard Grad-CAM: gradient of the target class's logit w.r.t.
the last spatial feature map, globally-average-pooled into per-channel
weights, applied as a weighted sum over the feature map, ReLU'd and
normalized into a 0..1 heatmap. Not class-specific in implementation --
works identically regardless of which of the 4 disease classes (or Healthy)
is predicted, so "ideally all classes" from the spec falls out for free.
"""
import base64
import io

import numpy as np
import torch
from PIL import Image
from PIL import ImageColor


def compute_gradcam(model, input_tensor: torch.Tensor, target_class_idx: int = None):
    """
    input_tensor: (1, C, H, W), already normalized, on the correct device.
    Returns (cam: np.ndarray shape (h_feat, w_feat) in [0, 1], target_class_idx: int).
    """
    model.eval()
    with torch.set_grad_enabled(True):
        features = model.forward_features(input_tensor)
        features.retain_grad()
        logits = model.forward_head(features)
        if target_class_idx is None:
            target_class_idx = int(logits.argmax(1).item())
        model.zero_grad(set_to_none=True)
        logits[0, target_class_idx].backward()

        grads = features.grad[0]       # (C, h, w)
        acts = features.detach()[0]    # (C, h, w)
        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(0))  # (h, w)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), target_class_idx


def overlay_png_base64(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
    """
    Upsamples `cam` (h_feat, w_feat, values in [0,1]) to `original_image`'s
    size, colorizes with a perceptually-uniform colormap, alpha-blends over
    the original image, and returns a base64-encoded PNG data string.
    """
    w, h = original_image.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0

    # Compact inferno-like ramp without a runtime matplotlib dependency.
    stops = [(0.0, "#000004"), (0.25, "#57106e"), (0.5, "#bc3754"),
             (0.75, "#f98e09"), (1.0, "#fcffa4")]
    colored = np.zeros((*cam_arr.shape, 3), dtype=np.float32)
    for (lo, lc), (hi, hc) in zip(stops[:-1], stops[1:]):
        mask = (cam_arr >= lo) & (cam_arr <= hi)
        t = np.clip((cam_arr - lo) / (hi - lo), 0, 1)[..., None]
        a = np.asarray(ImageColor.getrgb(lc), dtype=np.float32)
        b = np.asarray(ImageColor.getrgb(hc), dtype=np.float32)
        colored[mask] = (a + (b - a) * t[mask])
    colored = np.clip(colored, 0, 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored, mode="RGB")

    base = original_image.convert("RGB")
    blended = Image.blend(base, heatmap_img, alpha=alpha)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def gradcam_overlay_for_image(model, transform, original_image: Image.Image, device,
                               target_class_idx: int = None):
    """
    High-level entry point: preprocesses `original_image` the same way the
    classifier does, runs Grad-CAM, and returns a base64 PNG data-URI overlay
    ready to drop into an API response or an <img src="..."> tag.
    """
    x = transform(original_image.convert("RGB")).unsqueeze(0).to(device)
    cam, used_class_idx = compute_gradcam(model, x, target_class_idx)

    # Overlay on the same center-cropped view the model actually saw, not the
    # raw upload, so the heatmap is spatially honest about what the model
    # looked at (an overlay on the original uncropped photo would be
    # misaligned by the crop/resize the classifier applied).
    cropped_view = transform.transforms[0:2]  # Resize + CenterCrop, matching _LoadedModel's transform order
    view = original_image.convert("RGB")
    for t in cropped_view:
        view = t(view)

    overlay = overlay_png_base64(view, cam)
    return overlay, used_class_idx
