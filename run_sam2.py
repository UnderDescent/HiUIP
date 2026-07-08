# run_sam.py
# Usage: python run_sam.py path/to/screenshot.png
#
# Setup (one-time):
#   pip install transformers torch pillow numpy --break-system-packages
#
# First run will download the SAM model weights (~375MB for vit-base) automatically.
# Runs on CPU/MPS (Mac) or CUDA (4090) automatically -- whatever's available.

import sys
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import pipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_sam.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")
    print(f"Original image size: {image.size}")

    # SAM's automatic mode runs a dense grid of point prompts across the WHOLE
    # image -- on a tall full-page screenshot (common for these captures) this
    # can be thousands of points on CPU/MPS, which is very slow. Downscale so
    # the longest side is capped, to keep this testable before you have GPU time.
    MAX_DIM = 1280
    if max(image.size) > MAX_DIM:
        scale = MAX_DIM / max(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size)
        print(f"Downscaled to: {image.size} (scale factor {scale:.3f})")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # "mask-generation" = SAM's automatic mode: no prompt needed, it proposes
    # a full set of candidate region masks across the whole image on its own.
    generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=device)

    print("Running SAM (this can take 30s-2min on CPU, faster on GPU)...")
    # points_per_side controls grid density -- default is 32 (1024 points).
    # 16 (256 points) is ~4x fewer prompts, much faster, coarser results --
    # good enough for a first sanity check.
    outputs = generator(image, points_per_batch=64, points_per_side=16)

    masks = outputs["masks"]
    print(f"SAM proposed {len(masks)} regions")

    # Convert each mask to a bounding box for easy comparison against your DOM tree boxes
    boxes = []
    for mask in masks:
        mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
        mask = mask.squeeze()
        ys, xs = mask.nonzero()
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        boxes.append([x0, y0, x1 - x0, y1 - y0])  # [x, y, w, h] -- same format as your tree.json

    # Save boxes so you can load them into the same viewer you built for DOM trees
    import json
    with open("sam_boxes.json", "w") as f:
        json.dump(boxes, f)
    print("Saved sam_boxes.json")

    # Quick visual sanity check: draw all boxes on the image
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for (x, y, w, h) in boxes:
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    overlay.save("sam_overlay.png")
    print("Saved sam_overlay.png -- open this to see SAM's proposed regions")

if __name__ == "__main__":
    main()