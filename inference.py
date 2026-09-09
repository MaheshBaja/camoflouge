"""
Camouflaged Object Detection (COD) - Inference Engine & CLI Tool
Architecture: U-Net + ResNet34 (trained on COD10K-v3)
"""

import os
import time
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Union, Dict, Any, Optional

from model import load_model, get_default_device
from preprocessing import (
    preprocess_image,
    postprocess_prediction,
    create_highlight_overlay,
    TARGET_SIZE,
)


class CamouflageDetector:
    """
    High-performance detector class for Camouflaged Object Detection.
    Loads and holds the PyTorch model in memory for fast inference.
    """

    def __init__(
        self,
        checkpoint_path: str = "best_model.pth",
        device: Optional[str] = None
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_default_device()

        print(f"[*] Loading model from '{checkpoint_path}' on device '{self.device}'...")
        self.model, self.device, self.metadata = load_model(checkpoint_path, self.device)
        print(f"[OK] Model loaded successfully! (Best Checkpoint Epoch: {self.metadata.get('epoch', 'N/A')}, Val IoU: {self.metadata.get('best_iou', 'N/A'):.4f})")

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        threshold: float = 0.5,
        highlight_color: tuple = (0, 230, 118),
        alpha: float = 0.45
    ) -> Dict[str, Any]:
        """
        Runs complete inference pipeline on an input image.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess
        input_tensor, original_rgb, orig_shape = preprocess_image(image_input, target_size=TARGET_SIZE)
        input_tensor = input_tensor.to(self.device)

        # Step 2: Forward pass
        logits = self.model(input_tensor)

        # Step 3 & 4: Postprocess
        post_results = postprocess_prediction(
            logits,
            original_shape=orig_shape,
            threshold=threshold,
            highlight_color=highlight_color,
        )

        # Step 5: Highlight overlay
        binary_mask = post_results["binary_mask"]
        overlay_image = create_highlight_overlay(
            original_rgb,
            binary_mask,
            color=highlight_color,
            alpha=alpha,
            draw_contour=True
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "original_image": original_rgb,
            "binary_mask": binary_mask,
            "overlay_image": overlay_image,
            "heatmap": post_results["heatmap"],
            "prob_map": post_results["prob_map"],
            "inference_time_ms": elapsed_ms,
            "coverage_percent": post_results["coverage_percent"],
            "camouflaged_pixels": post_results["camouflaged_pixels"],
            "total_pixels": post_results["total_pixels"],
            "max_confidence": post_results["max_confidence"],
            "mean_confidence": post_results["mean_confidence"],
            "original_shape": orig_shape,
        }


def save_prediction_results(
    results: Dict[str, Any],
    output_dir: str,
    base_name: str,
    save_composite: bool = True
) -> Dict[str, str]:
    """
    Saves the prediction outputs to disk:
      - base_name_mask.png (Binary mask)
      - base_name_overlay.png (Highlighted camouflage)
      - base_name_heatmap.png (Confidence heatmap)
      - base_name_comparison.png (Side-by-side comparison)
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    # 1. Binary Mask (Grayscale)
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, results["binary_mask"])
    saved_paths["mask"] = mask_path

    # 2. Highlighted Overlay (RGB to BGR)
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(results["overlay_image"], cv2.COLOR_RGB2BGR))
    saved_paths["overlay"] = overlay_path

    # 3. Heatmap
    heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    cv2.imwrite(heatmap_path, cv2.cvtColor(results["heatmap"], cv2.COLOR_RGB2BGR))
    saved_paths["heatmap"] = heatmap_path

    # 4. Side-by-side composite
    if save_composite:
        orig = results["original_image"]
        mask_rgb = cv2.cvtColor(results["binary_mask"], cv2.COLOR_GRAY2RGB)
        overlay = results["overlay_image"]
        heatmap = results["heatmap"]

        composite_rgb = np.hstack([orig, mask_rgb, overlay, heatmap])
        composite_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        cv2.imwrite(composite_path, cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))
        saved_paths["composite"] = composite_path

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Camouflaged Object Detection Inference (U-Net + ResNet34)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file for detection."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to a folder of images for batch detection."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
        help="Path to trained model checkpoint (default: best_model.pth)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save predicted masks and highlighted images (default: outputs/)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for binary segmentation (default: 0.5)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device for computation: 'cuda' or 'cpu' (default: auto-detect)."
    )

    args = parser.parse_args()

    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            return
        image_paths.append(args.image)
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[ERROR] Input directory not found: {args.input_dir}")
            return
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        for f in sorted(os.listdir(args.input_dir)):
            if os.path.splitext(f.lower())[1] in valid_exts:
                image_paths.append(os.path.join(args.input_dir, f))
    else:
        if os.path.exists("test_images") and len(os.listdir("test_images")) > 0:
            print("[*] No --image or --input_dir provided. Running on 'test_images/' folder...")
            for f in sorted(os.listdir("test_images")):
                if os.path.splitext(f.lower())[1] in {".jpg", ".jpeg", ".png", ".webp"}:
                    image_paths.append(os.path.join("test_images", f))
        else:
            print("[ERROR] Please provide --image <path> or --input_dir <path>.")
            return

    if not image_paths:
        print("[!] No valid images found to process.")
        return

    detector = CamouflageDetector(checkpoint_path=args.model_path, device=args.device)

    print(f"\n{'='*70}")
    print(f"Camouflaged Object Detection - Inference Pipeline")
    print(f"Processing {len(image_paths)} image(s) | Threshold: {args.threshold} | Output: {args.output_dir}")
    print(f"{'='*70}\n")

    for idx, img_path in enumerate(image_paths, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx}/{len(image_paths)}] Processing: {img_path} ...", end=" ", flush=True)

        results = detector.predict(img_path, threshold=args.threshold)
        saved = save_prediction_results(results, args.output_dir, stem)

        print(f"Done in {results['inference_time_ms']:.1f}ms")
        print(f"    - Original Resolution : {results['original_shape'][1]}x{results['original_shape'][0]}")
        print(f"    - Camouflage Area     : {results['coverage_percent']:.2f}% of image")
        print(f"    - Max Confidence      : {results['max_confidence']:.4f}")
        print(f"    - Outputs Saved       : {saved['mask']}, {saved['overlay']}")

    print(f"\n[OK] All results successfully saved to '{os.path.abspath(args.output_dir)}'.\n")


if __name__ == "__main__":
    main()
