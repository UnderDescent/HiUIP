# build_training_pairs.py
# Usage: python build_training_pairs.py out/ train_sites.json train_pairs.jsonl
#        python build_training_pairs.py out/ eval_sites.json eval_pairs.jsonl
#
# For every node in every tree, emits one training example:
#   { image_path, point: [x,y], target_box: [x,y,w,h], depth }
#
# point is a random point inside the box (this becomes the prompt SAM sees).
# target_box is what SAM should learn to output when prompted at that point.
# This is the raw material for SAM's decoder fine-tuning loop.

import sys
import os
import json
import random
import struct

IMAGE_NAMES = ("screenshot.png", "screenshot.jpeg", "screenshot.jpg")

def find_screenshot(site_dir):
    for name in IMAGE_NAMES:
        candidate = os.path.join(site_dir, name)
        if os.path.exists(candidate):
            return candidate
    return None

def image_size(path):
    with open(path, "rb") as f:
        header = f.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", header[16:24])
        if header[:3] == b"\xff\xd8\xff":
            f.seek(2)
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if marker in (b"\xd8", b"\xd9"):
                    continue
                length_bytes = f.read(2)
                if len(length_bytes) != 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                if marker in {
                    b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6",
                    b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce",
                    b"\xcf",
                }:
                    data = f.read(5)
                    if len(data) != 5:
                        break
                    height, width = struct.unpack(">HH", data[1:5])
                    return width, height
                f.seek(length - 2, os.SEEK_CUR)
    raise ValueError(f"Unsupported or corrupt image: {path}")

def collect_pairs(node, image_path, image_w, image_h, depth, pairs):
    if node is None:
        return
    x, y, w, h = node["bbox"]
    if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= image_w and y + h <= image_h:
        # Random point inside the box -- avoids the model overfitting to
        # always seeing the exact center, matches how SAM is prompted in
        # practice (a rough click anywhere inside the target).
        px = x + random.uniform(0.2, 0.8) * w
        py = y + random.uniform(0.2, 0.8) * h
        pairs.append({
            "image_path": image_path,
            "point": [round(px, 1), round(py, 1)],
            "target_box": [x, y, w, h],
            "depth": depth,
            "tag": node.get("tag"),
        })
    for child in node.get("children", []):
        collect_pairs(child, image_path, image_w, image_h, depth + 1, pairs)

def main():
    if len(sys.argv) < 4:
        print("Usage: python build_training_pairs.py <out_dir> <sites_json> <output.jsonl>")
        sys.exit(1)

    out_dir, sites_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(sites_file) as f:
        sites = json.load(f)

    random.seed(42)
    all_pairs = []
    skipped = 0

    for site in sites:
        site_dir = os.path.join(out_dir, site)
        tree_path = os.path.join(site_dir, "tree.json")
        image_path = find_screenshot(site_dir)

        if not os.path.exists(tree_path) or image_path is None:
            skipped += 1
            continue

        with open(tree_path) as f:
            tree = json.load(f)

        if tree is None:
            skipped += 1
            continue

        image_w, image_h = image_size(image_path)
        pairs = []
        collect_pairs(tree, os.path.abspath(image_path), image_w, image_h, 0, pairs)
        all_pairs.extend(pairs)
        print(f"{site}: {len(pairs)} pairs")

    with open(output_file, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nTotal: {len(all_pairs)} training pairs from {len(sites) - skipped} sites ({skipped} skipped)")
    print(f"Written to {output_file}")

if __name__ == "__main__":
    main()
