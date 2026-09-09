"""
Camouflaged Object Detection (COD) - Preprocessing & Postprocessing Pipeline
Matches exact training pipeline:
  - Input resolution: 352 x 352
  - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Sigmoid activation + thresholding + clean overlay visualization
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, Dict, Any


# Training constants
TARGET_SIZE = (352, 352)  # (width, height)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_as_rgb(image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Converts various image input formats into an RGB uint8 NumPy array.
    """
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                rgb = image_input.copy()
            else:
                raise ValueError(f"Unsupported number of channels: {image_input.shape[2]}")
        else:
            raise ValueError(f"Unsupported image shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    return rgb.astype(np.uint8)


def preprocess_image(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Preprocesses an input image according to the training specification:
      1. Load and convert to RGB
      2. Record original dimensions (H, W)
      3. Resize to target size (352 x 352) using bilinear interpolation
      4. Normalize pixel values to [0, 1]
      5. Standardize with ImageNet mean and std
      6. Convert to PyTorch Tensor of shape (1, 3, 352, 352)

    Returns:
        tensor: torch.Tensor of shape (1, 3, 352, 352), float32
        original_rgb: np.ndarray of shape (H, W, 3), uint8
        original_shape: tuple of (height, width)
    """
    original_rgb = load_image_as_rgb(image_input)
    orig_h, orig_w = original_rgb.shape[:2]

    # Resize to 352x352
    resized_rgb = cv2.resize(original_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0.0, 1.0]
    normalized = resized_rgb.astype(np.float32) / 255.0

    # ImageNet standardization
    standardized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose from (H, W, C) to (C, H, W)
    chw = np.transpose(standardized, (2, 0, 1))

    # Convert to torch tensor with batch dimension (1, C, H, W)
    tensor = torch.from_numpy(chw).unsqueeze(0).float()

    return tensor, original_rgb, (orig_h, orig_w)


def create_highlight_overlay(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 230, 118),  # Vibrant Emerald Green (R, G, B)
    alpha: float = 0.45,
    draw_contour: bool = True
) -> np.ndarray:
    """
    Creates an overlay with semi-transparent color highlighting the camouflaged object
    and a crisp boundary contour outline.
    """
    overlay = original_rgb.copy()
    mask_bool = binary_mask > 0

    if not np.any(mask_bool):
        return overlay

    # Create solid color mask
    color_layer = np.zeros_like(original_rgb, dtype=np.uint8)
    color_layer[mask_bool] = color

    # Blend colored region with original image
    blended = cv2.addWeighted(color_layer, alpha, original_rgb, 1.0 - alpha, 0)
    overlay[mask_bool] = blended[mask_bool]

    # Draw crisp contour boundary
    if draw_contour:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), 4)
        cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def create_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """
    Generates a Jet colormap heatmap from the probability map (0.0 to 1.0).
    """
    prob_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def postprocess_prediction(
    model_output: Union[torch.Tensor, np.ndarray],
    original_shape: Tuple[int, int],
    threshold: float = 0.5,
    highlight_color: Tuple[int, int, int] = (0, 230, 118)
) -> Dict[str, Any]:
    """
    Postprocesses the raw model logits into actionable outputs.
    """
    orig_h, orig_w = original_shape

    # Apply sigmoid to logits
    if isinstance(model_output, torch.Tensor):
        probs = torch.sigmoid(model_output).detach().cpu().squeeze().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-model_output.squeeze()))

    if probs.ndim != 2:
        probs = probs.squeeze()

    # Resize probability map back to original image resolution
    prob_map_orig = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    prob_map_orig = np.clip(prob_map_orig, 0.0, 1.0)

    # Threshold binary mask
    binary_mask = (prob_map_orig >= threshold).astype(np.uint8) * 255

    # Compute metrics
    total_pixels = orig_h * orig_w
    camouflaged_pixels = int(np.sum(binary_mask > 0))
    coverage_percent = (camouflaged_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    detected_probs = prob_map_orig[binary_mask > 0] if camouflaged_pixels > 0 else prob_map_orig
    max_prob = float(np.max(prob_map_orig))
    mean_prob = float(np.mean(detected_probs))

    # Heatmap
    heatmap = create_heatmap(prob_map_orig)

    return {
        "prob_map": prob_map_orig,
        "binary_mask": binary_mask,
        "heatmap": heatmap,
        "coverage_percent": coverage_percent,
        "camouflaged_pixels": camouflaged_pixels,
        "total_pixels": total_pixels,
        "max_confidence": max_prob,
        "mean_confidence": mean_prob,
    }
