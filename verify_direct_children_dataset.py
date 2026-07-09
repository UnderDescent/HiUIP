#!/usr/bin/env python3
"""Validate parent-crop -> direct-child-box JSONL records."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def box_area(box):
    return box[2] * box[3]


def inside_box(box, width, height):
    x, y, w, h = box
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height


def load_rows(path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def validate(path):
    errors = []
    warnings = []
    counts = Counter()
    image_sizes = {}

    for line_no, row in load_rows(path):
        label = f"{path}:{line_no}"
        image_path = Path(row.get("image_path", ""))
        if not image_path.exists():
            errors.append(f"{label}: missing image {image_path}")
            continue

        if image_path not in image_sizes:
            with Image.open(image_path) as image:
                image_sizes[image_path] = image.size
        image_w, image_h = image_sizes[image_path]

        crop_box = row.get("crop_box")
        if not inside_box(crop_box, image_w, image_h):
            errors.append(f"{label}: crop_box {crop_box} outside image {(image_w, image_h)}")
            continue

        crop_w, crop_h = crop_box[2], crop_box[3]
        if row.get("crop_size") != [crop_w, crop_h]:
            errors.append(f"{label}: crop_size {row.get('crop_size')} does not match crop_box {crop_box}")

        child_boxes = row.get("child_boxes")
        if not isinstance(child_boxes, list) or len(child_boxes) < 2:
            errors.append(f"{label}: child_boxes must contain at least two boxes")
            continue

        for child_index, child_box in enumerate(child_boxes):
            child_label = f"{label}: child {child_index}"
            if not inside_box(child_box, crop_w, crop_h):
                errors.append(f"{child_label}: child_box {child_box} outside crop {(crop_w, crop_h)}")
                continue
            if box_area(child_box) < 64:
                warnings.append(f"{child_label}: tiny child_box {child_box}")

        for child_index, child in enumerate(row.get("children", [])):
            if child.get("box") != child_boxes[child_index]:
                errors.append(f"{label}: children[{child_index}].box does not match child_boxes")

        counts["rows"] += 1
        counts["children"] += len(child_boxes)
        counts[row.get("site", "unknown")] += 1

    print(f"\n=== {path} ===")
    print(f"Rows: {counts['rows']}")
    print(f"Child boxes: {counts['children']}")
    print("Top sites:", ", ".join(f"{site}={count}" for site, count in counts.most_common(8) if site not in {"rows", "children"}))

    print("\nErrors:")
    if errors:
        for error in errors[:25]:
            print(f"- {error}")
        if len(errors) > 25:
            print(f"... and {len(errors) - 25} more")
    else:
        print("None.")

    print("\nWarnings:")
    if warnings:
        for warning in warnings[:25]:
            print(f"- {warning}")
        if len(warnings) > 25:
            print(f"... and {len(warnings) - 25} more")
    else:
        print("None.")

    return len(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    args = parser.parse_args()

    error_count = 0
    for path in args.jsonl:
        error_count += validate(path)
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
