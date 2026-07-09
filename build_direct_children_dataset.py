#!/usr/bin/env python3
"""Build parent-crop -> direct-child-box examples from scraped UI trees.

Usage:
  python build_direct_children_dataset.py out/ train_sites.json train_direct_children.jsonl

Each output row describes one training/inference situation:
  - crop_box: the parent element region to crop from the screenshot
  - child_boxes: the direct child boxes inside that crop, relative to crop origin

Tiny/hidden/out-of-frame boxes are removed, and obvious one-child wrapper chains
are collapsed so a wrapper with one rich child does not become a useless target.
"""

import argparse
import json
import os
import struct
from pathlib import Path


IMAGE_NAMES = ("screenshot.png", "screenshot.jpeg", "screenshot.jpg")
WRAPPER_TAGS = {"body", "html", "tbody"}


def find_screenshot(site_dir):
    for name in IMAGE_NAMES:
        candidate = site_dir / name
        if candidate.exists():
            return candidate
    return None


def image_size(path):
    with path.open("rb") as f:
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


def load_scale(site_dir, image_w, image_h):
    meta_path = site_dir / "meta.json"
    if not meta_path.exists():
        return 1.0, 1.0, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    viewport = meta.get("viewport")
    if (
        isinstance(viewport, list)
        and len(viewport) == 2
        and viewport[0] > 0
        and viewport[1] > 0
    ):
        return image_w / viewport[0], image_h / viewport[1], viewport
    return 1.0, 1.0, viewport


def valid_box(node, image_w, image_h, scale_x=1.0, scale_y=1.0):
    if not isinstance(node, dict):
        return None
    box = node.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    x, y, w, h = box
    x *= scale_x
    y *= scale_y
    w *= scale_x
    h *= scale_y
    if w <= 0 or h <= 0:
        return None
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_w, x + w)
    y2 = min(image_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1), round(y1), round(x2 - x1), round(y2 - y1)]


def area(box):
    return box[2] * box[3]


def intersection(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union else 0


def visible_children(node, image_w, image_h, min_box_area, scale_x, scale_y):
    children = []
    for child in node.get("children", []):
        box = valid_box(child, image_w, image_h, scale_x, scale_y)
        if box and area(box) >= min_box_area:
            children.append(child)
    return children


def child_payload(node, image_w, image_h, scale_x, scale_y):
    return {
        "tag": node.get("tag"),
        "role": node.get("role"),
        "id": node.get("id"),
        "classes": node.get("classes", [])[:5],
        "box": valid_box(node, image_w, image_h, scale_x, scale_y),
    }


def collapse_child_wrapper(node, image_w, image_h, min_box_area, scale_x, scale_y):
    current = node
    collapsed = 0
    while True:
        children = visible_children(current, image_w, image_h, min_box_area, scale_x, scale_y)
        if len(children) != 1:
            return current, collapsed
        current_box = valid_box(current, image_w, image_h, scale_x, scale_y)
        child_box = valid_box(children[0], image_w, image_h, scale_x, scale_y)
        if not current_box or not child_box:
            return current, collapsed
        same_visual_region = iou(current_box, child_box) >= 0.85
        tag_is_wrapper = (current.get("tag") or "").lower() in WRAPPER_TAGS
        if not same_visual_region and not tag_is_wrapper:
            return current, collapsed
        current = children[0]
        collapsed += 1


def effective_children(node, image_w, image_h, min_box_area, scale_x, scale_y):
    current = node
    parent_collapsed = 0
    while True:
        children = visible_children(current, image_w, image_h, min_box_area, scale_x, scale_y)
        if len(children) != 1:
            break
        grandkids = visible_children(children[0], image_w, image_h, min_box_area, scale_x, scale_y)
        if not grandkids:
            break
        current = children[0]
        parent_collapsed += 1

    normalized = []
    child_collapsed = 0
    for child in visible_children(current, image_w, image_h, min_box_area, scale_x, scale_y):
        normalized_child, collapsed = collapse_child_wrapper(
            child, image_w, image_h, min_box_area, scale_x, scale_y
        )
        normalized.append(normalized_child)
        child_collapsed += collapsed
    return current, normalized, parent_collapsed, child_collapsed


def relative_box(child_box, parent_box):
    px, py, pw, ph = parent_box
    cx, cy, cw, ch = child_box
    x1 = max(px, cx)
    y1 = max(py, cy)
    x2 = min(px + pw, cx + cw)
    y2 = min(py + ph, cy + ch)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1 - px), round(y1 - py), round(x2 - x1), round(y2 - y1)]


def collect_examples(node, image_path, image_w, image_h, site, depth, args, rows, stats, scale_x, scale_y, viewport):
    parent_box = valid_box(node, image_w, image_h, scale_x, scale_y)
    if parent_box and area(parent_box) >= args.min_parent_area:
        normalized_parent, children, parent_collapsed, child_collapsed = effective_children(
            node, image_w, image_h, args.min_child_area, scale_x, scale_y
        )
        normalized_box = valid_box(normalized_parent, image_w, image_h, scale_x, scale_y)
        crop_fits_limits = (
            normalized_box
            and normalized_box[2] <= args.max_crop_width
            and normalized_box[3] <= args.max_crop_height
            and area(normalized_box) <= args.max_crop_area
        )
        if crop_fits_limits and len(children) >= args.min_children:
            child_boxes = []
            child_meta = []
            for child in children:
                child_box = valid_box(child, image_w, image_h, scale_x, scale_y)
                rel = relative_box(child_box, normalized_box) if child_box else None
                if rel and area(rel) >= args.min_child_area:
                    child_boxes.append(rel)
                    meta = child_payload(child, image_w, image_h, scale_x, scale_y)
                    meta["box"] = rel
                    child_meta.append(meta)

            if len(child_boxes) >= args.min_children:
                rows.append({
                    "site": site,
                    "image_path": str(image_path.resolve()),
                    "crop_box": normalized_box,
                    "crop_size": [normalized_box[2], normalized_box[3]],
                    "image_size": [image_w, image_h],
                    "source_viewport": viewport,
                    "box_scale": [scale_x, scale_y],
                    "child_boxes": child_boxes,
                    "child_count": len(child_boxes),
                    "parent": child_payload(normalized_parent, image_w, image_h, scale_x, scale_y),
                    "children": child_meta,
                    "depth": depth + parent_collapsed,
                    "collapsed_parent_levels": parent_collapsed,
                    "collapsed_child_levels": child_collapsed,
                })
                stats["rows"] += 1
                stats["child_boxes"] += len(child_boxes)
                stats["collapsed_parent_levels"] += parent_collapsed
                stats["collapsed_child_levels"] += child_collapsed

    for child in node.get("children", []):
        if isinstance(child, dict):
            collect_examples(
                child, image_path, image_w, image_h, site, depth + 1,
                args, rows, stats, scale_x, scale_y, viewport
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("sites_json", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--min-parent-area", type=int, default=1600)
    parser.add_argument("--min-child-area", type=int, default=64)
    parser.add_argument("--min-children", type=int, default=2)
    parser.add_argument("--max-crop-width", type=int, default=3000)
    parser.add_argument("--max-crop-height", type=int, default=1800)
    parser.add_argument("--max-crop-area", type=int, default=4500000)
    args = parser.parse_args()

    sites = json.loads(args.sites_json.read_text(encoding="utf-8"))
    rows = []
    stats = {
        "rows": 0,
        "child_boxes": 0,
        "collapsed_parent_levels": 0,
        "collapsed_child_levels": 0,
    }
    skipped = 0

    for site in sites:
        site_dir = args.out_dir / site
        tree_path = site_dir / "tree.json"
        image_path = find_screenshot(site_dir)
        if not tree_path.exists() or image_path is None:
            skipped += 1
            continue
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        image_w, image_h = image_size(image_path)
        scale_x, scale_y, viewport = load_scale(site_dir, image_w, image_h)
        before = len(rows)
        collect_examples(
            tree, image_path, image_w, image_h, site, 0,
            args, rows, stats, scale_x, scale_y, viewport
        )
        print(
            f"{site}: {len(rows) - before} parent-crop examples "
            f"(scale {scale_x:.3f} x {scale_y:.3f})"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTotal examples: {len(rows)} from {len(sites) - skipped} sites ({skipped} skipped)")
    print(f"Total child boxes: {stats['child_boxes']}")
    print(f"Collapsed parent levels: {stats['collapsed_parent_levels']}")
    print(f"Collapsed child levels: {stats['collapsed_child_levels']}")
    print(f"Written to {args.output_jsonl}")


if __name__ == "__main__":
    main()
