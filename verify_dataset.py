#!/usr/bin/env python3
"""Validate scraped UI trees and generated SAM training pairs.

Usage:
  python3 verify_dataset.py
  python3 verify_dataset.py --out out --train train_pairs.jsonl --eval eval_pairs.jsonl
"""

import argparse
import json
import math
import os
import struct
import sys
from collections import Counter
from pathlib import Path


IMAGE_NAMES = ("screenshot.png", "screenshot.jpeg", "screenshot.jpg")


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def print_block(self, title, items, limit=25):
        print(f"\n=== {title} ===")
        if not items:
            print("None.")
            return
        for item in items[:limit]:
            print(f"- {item}")
        if len(items) > limit:
            print(f"... and {len(items) - limit} more")


def load_json(path, report, label):
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        report.error(f"{label} is missing: {path}")
    except json.JSONDecodeError as exc:
        report.error(f"{label} is invalid JSON: {path}:{exc.lineno}:{exc.colno}")
    return None


def load_jsonl(path, report, label):
    rows = []
    try:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    report.warn(f"{label}:{line_no} is blank")
                    continue
                try:
                    rows.append((line_no, json.loads(line)))
                except json.JSONDecodeError as exc:
                    report.error(f"{label}:{line_no} is invalid JSON: {exc.msg}")
    except FileNotFoundError:
        report.error(f"{label} is missing: {path}")
    return rows


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
    raise ValueError("unsupported or corrupt image")


def site_from_image_path(path):
    parent = Path(path).parent
    return parent.name


def find_screenshot(site_dir):
    for name in IMAGE_NAMES:
        candidate = site_dir / name
        if candidate.exists():
            return candidate
    return None


def valid_box(value):
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        return None
    return value


def count_tree_pairs(node, report, tree_path, dimensions=None, depth=0):
    if node is None:
        return 0, 0
    if not isinstance(node, dict):
        report.error(f"{tree_path}: node at depth {depth} is not an object")
        return 0, 0

    count = 0
    max_depth = depth
    box = valid_box(node.get("bbox"))
    if box is None:
        report.error(f"{tree_path}: node at depth {depth} has malformed bbox")
    else:
        x, y, width, height = box
        fits_image = (
            dimensions is None
            or (x >= 0 and y >= 0 and x + width <= dimensions[0] and y + height <= dimensions[1])
        )
        if width > 0 and height > 0 and fits_image:
            count += 1

    children = node.get("children", [])
    if not isinstance(children, list):
        report.error(f"{tree_path}: node at depth {depth} has non-list children")
        return count, max_depth

    for child in children:
        child_count, child_depth = count_tree_pairs(child, report, tree_path, dimensions, depth + 1)
        count += child_count
        max_depth = max(max_depth, child_depth)
    return count, max_depth


def inspect_sites(out_dir, train_sites, eval_sites, report):
    print("\n=== Scraped site folders ===")
    site_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.exists() else []
    print(f"Found {len(site_dirs)} site folder(s) under {out_dir}")

    split_sites = set(train_sites) | set(eval_sites)
    actual_sites = {p.name for p in site_dirs}
    for site in sorted(split_sites - actual_sites):
        report.error(f"split references missing site folder: {site}")
    for site in sorted(actual_sites - split_sites):
        report.warn(f"site folder is not in train/eval split: {site}")

    info = {}
    for site_dir in site_dirs:
        site = site_dir.name
        tree_path = site_dir / "tree.json"
        screenshot = find_screenshot(site_dir)
        meta_path = site_dir / "meta.json"

        if screenshot is None:
            report.error(f"{site}: missing screenshot file")
            dimensions = None
        else:
            try:
                dimensions = image_size(screenshot)
            except ValueError as exc:
                report.error(f"{site}: cannot read {screenshot.name}: {exc}")
                dimensions = None

        tree = load_json(tree_path, report, f"{site}/tree.json")
        expected_pairs = 0
        max_depth = 0
        if tree is not None:
            expected_pairs, max_depth = count_tree_pairs(tree, report, tree_path, dimensions)

        meta = load_json(meta_path, report, f"{site}/meta.json") if meta_path.exists() else None
        if meta:
            node_count = meta.get("nodeCount")
            if isinstance(node_count, int) and expected_pairs and node_count < expected_pairs:
                report.warn(f"{site}: meta nodeCount {node_count} is smaller than valid bbox count {expected_pairs}")

        info[site] = {
            "tree_path": tree_path,
            "screenshot": screenshot,
            "dimensions": dimensions,
            "expected_pairs": expected_pairs,
            "max_depth": max_depth,
        }

    return info


def inspect_pairs(rows, label, allowed_sites, site_info, report):
    print(f"\n=== {label} pairs ===")
    print(f"Loaded {len(rows)} record(s)")

    by_site = Counter()
    tags = Counter()
    depths = Counter()
    duplicate_keys = Counter()
    areas = []
    seen_records = Counter()

    for line_no, row in rows:
        where = f"{label}:{line_no}"
        if not isinstance(row, dict):
            report.error(f"{where}: record is not an object")
            continue

        image_path = row.get("image_path")
        point = row.get("point")
        box = valid_box(row.get("target_box"))
        depth = row.get("depth")
        tag = row.get("tag")

        if not isinstance(image_path, str) or not image_path:
            report.error(f"{where}: missing image_path")
            continue

        image = Path(image_path)
        site = site_from_image_path(image)
        by_site[site] += 1
        tags[tag or "(none)"] += 1

        if site not in allowed_sites:
            report.error(f"{where}: site {site} is not in the {label} split")
        if not image.exists():
            report.error(f"{where}: image does not exist: {image}")
            dimensions = None
        else:
            dimensions = site_info.get(site, {}).get("dimensions")
            expected = site_info.get(site, {}).get("screenshot")
            if expected and image.resolve() != expected.resolve():
                report.error(f"{where}: image_path points outside the site's screenshot: {image}")

        if box is None:
            report.error(f"{where}: malformed target_box {row.get('target_box')!r}")
            continue
        x, y, width, height = box
        if width <= 0 or height <= 0:
            report.error(f"{where}: non-positive target_box size {box}")
            continue
        if x < 0 or y < 0:
            report.error(f"{where}: target_box has negative origin {box}")
        if dimensions and (x + width > dimensions[0] + 1 or y + height > dimensions[1] + 1):
            report.error(f"{where}: target_box {box} exceeds image bounds {dimensions}")
        areas.append(width * height)

        if not isinstance(point, list) or len(point) != 2:
            report.error(f"{where}: malformed point {point!r}")
        elif not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            report.error(f"{where}: point contains non-finite values {point!r}")
        else:
            px, py = point
            if not (x <= px <= x + width and y <= py <= y + height):
                report.error(f"{where}: point {point} falls outside target_box {box}")
            if dimensions and not (0 <= px <= dimensions[0] and 0 <= py <= dimensions[1]):
                report.error(f"{where}: point {point} falls outside image bounds {dimensions}")

        if not isinstance(depth, int) or depth < 0:
            report.error(f"{where}: invalid depth {depth!r}")
        else:
            depths[depth] += 1

        duplicate_keys[(str(image), tuple(box))] += 1
        seen_records[json.dumps(row, sort_keys=True)] += 1

    for site, expected in sorted((s, site_info.get(s, {}).get("expected_pairs", 0)) for s in allowed_sites):
        actual = by_site[site]
        if expected and actual != expected:
            report.error(f"{label}: site {site} has {actual} pairs, expected {expected} from tree.json")

    exact_box_dupes = sum(1 for count in duplicate_keys.values() if count > 1)
    exact_record_dupes = sum(1 for count in seen_records.values() if count > 1)
    if exact_box_dupes:
        report.warn(f"{label}: {exact_box_dupes} duplicate (image_path, target_box) key(s)")
    if exact_record_dupes:
        report.warn(f"{label}: {exact_record_dupes} completely duplicate record(s)")

    if tags:
        print("Top tags:", ", ".join(f"{tag}={count}" for tag, count in tags.most_common(8)))
    if depths:
        print("Depth range:", min(depths), "to", max(depths))
    if areas:
        areas.sort()
        print(
            "Box area px^2:",
            f"min={areas[0]:.0f}",
            f"median={areas[len(areas)//2]:.0f}",
            f"p95={areas[int((len(areas)-1)*0.95)]:.0f}",
            f"max={areas[-1]:.0f}",
        )

    return by_site


def main():
    parser = argparse.ArgumentParser(description="Confirm whether the HUIP dataset is clean.")
    parser.add_argument("--out", default="out", type=Path)
    parser.add_argument("--train-sites", default="train_sites.json", type=Path)
    parser.add_argument("--eval-sites", default="eval_sites.json", type=Path)
    parser.add_argument("--train", default="train_pairs.jsonl", type=Path)
    parser.add_argument("--eval", default="eval_pairs.jsonl", type=Path)
    args = parser.parse_args()

    report = Report()
    train_sites = load_json(args.train_sites, report, "train sites") or []
    eval_sites = load_json(args.eval_sites, report, "eval sites") or []
    if not isinstance(train_sites, list):
        report.error(f"{args.train_sites} must contain a JSON list")
        train_sites = []
    if not isinstance(eval_sites, list):
        report.error(f"{args.eval_sites} must contain a JSON list")
        eval_sites = []

    train_set = set(train_sites)
    eval_set = set(eval_sites)
    overlap = train_set & eval_set
    if overlap:
        report.error(f"train/eval split overlaps on {len(overlap)} site(s): {sorted(overlap)}")
    if len(train_set) != len(train_sites):
        report.error("train_sites.json contains duplicate site names")
    if len(eval_set) != len(eval_sites):
        report.error("eval_sites.json contains duplicate site names")

    site_info = inspect_sites(args.out, train_set, eval_set, report)
    train_rows = load_jsonl(args.train, report, "train")
    eval_rows = load_jsonl(args.eval, report, "eval")
    train_counts = inspect_pairs(train_rows, "train", train_set, site_info, report)
    eval_counts = inspect_pairs(eval_rows, "eval", eval_set, site_info, report)

    train_images = {row.get("image_path") for _, row in train_rows if isinstance(row, dict)}
    eval_images = {row.get("image_path") for _, row in eval_rows if isinstance(row, dict)}
    image_overlap = train_images & eval_images
    if image_overlap:
        report.error(f"train/eval JSONL image_path leakage: {len(image_overlap)} overlapping image(s)")

    print("\n=== Split summary ===")
    print(f"Train sites: {len(train_set)} | records: {sum(train_counts.values())}")
    print(f"Eval sites: {len(eval_set)} | records: {sum(eval_counts.values())}")

    report.print_block("Errors", report.errors)
    report.print_block("Warnings", report.warnings)

    if report.errors:
        print(f"\nNOT CLEAN: found {len(report.errors)} error(s) and {len(report.warnings)} warning(s).")
        return 1

    print(f"\nCLEAN: no blocking errors found ({len(report.warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
