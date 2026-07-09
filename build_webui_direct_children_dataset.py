#!/usr/bin/env python3
"""Convert WebUI samples into parent-crop -> direct-child-box JSONL rows.

WebUI stores hierarchy in *-axtree.json.gz and element boxes in *-bb.json.gz.
Accessibility nodes are linked to boxes through backendDOMNodeId.
"""

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SKIP_CHILD_ROLES = {"LineBreak"}
WRAPPER_ROLES = {"none", "generic", "group", "section"}


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def role_value(node):
    role = node.get("role") if isinstance(node, dict) else None
    if isinstance(role, dict):
        return role.get("value")
    return role


def node_name(node):
    name = node.get("name") if isinstance(node, dict) else None
    if isinstance(name, dict):
        return name.get("value")
    return name


def box_area(box):
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
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0


def valid_box(node, boxes, image_w, image_h):
    backend_id = node.get("backendDOMNodeId")
    if backend_id is None:
        return None
    raw = boxes.get(str(backend_id))
    if not raw:
        return None

    x = raw.get("x")
    y = raw.get("y")
    w = raw.get("width")
    h = raw.get("height")
    if None in (x, y, w, h) or w <= 0 or h <= 0:
        return None

    x1 = max(0, float(x))
    y1 = max(0, float(y))
    x2 = min(image_w, float(x) + float(w))
    y2 = min(image_h, float(y) + float(h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1), round(y1), round(x2 - x1), round(y2 - y1)]


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


def child_ids(node):
    ids = node.get("childIds", [])
    return ids if isinstance(ids, list) else []


def visible_child_nodes(node, nodes_by_id, boxes, image_w, image_h, min_area):
    """Return visible semantic children, expanding ignored/boxless wrappers."""
    out = []
    stack = list(reversed(child_ids(node)))
    while stack:
        child_id = stack.pop()
        child = nodes_by_id.get(str(child_id))
        if not child:
            continue

        role = role_value(child)
        box = valid_box(child, boxes, image_w, image_h)
        invisible = box is None or box_area(box) < min_area
        should_expand = child.get("ignored") or invisible or role in SKIP_CHILD_ROLES

        if should_expand and child_ids(child):
            stack.extend(reversed(child_ids(child)))
            continue
        if not invisible and role not in SKIP_CHILD_ROLES:
            out.append(child)
    return out


def collapse_one_child_wrapper(node, nodes_by_id, boxes, image_w, image_h, min_area):
    current = node
    collapsed = 0
    while True:
        children = visible_child_nodes(current, nodes_by_id, boxes, image_w, image_h, min_area)
        if len(children) != 1:
            return current, collapsed

        current_box = valid_box(current, boxes, image_w, image_h)
        child_box = valid_box(children[0], boxes, image_w, image_h)
        if not current_box or not child_box:
            return current, collapsed

        same_region = iou(current_box, child_box) >= 0.85
        wrapper_role = role_value(current) in WRAPPER_ROLES or current.get("ignored")
        if not same_region and not wrapper_role:
            return current, collapsed

        current = children[0]
        collapsed += 1


def effective_children(node, nodes_by_id, boxes, image_w, image_h, min_area):
    current = node
    parent_collapsed = 0
    while True:
        children = visible_child_nodes(current, nodes_by_id, boxes, image_w, image_h, min_area)
        if len(children) != 1:
            break
        grandkids = visible_child_nodes(children[0], nodes_by_id, boxes, image_w, image_h, min_area)
        if not grandkids:
            break
        current = children[0]
        parent_collapsed += 1

    normalized = []
    child_collapsed = 0
    for child in visible_child_nodes(current, nodes_by_id, boxes, image_w, image_h, min_area):
        normalized_child, collapsed = collapse_one_child_wrapper(
            child, nodes_by_id, boxes, image_w, image_h, min_area
        )
        normalized.append(normalized_child)
        child_collapsed += collapsed
    return current, normalized, parent_collapsed, child_collapsed


def payload(node, box):
    return {
        "tag": None,
        "role": role_value(node),
        "id": str(node.get("nodeId")),
        "classes": [],
        "name": (node_name(node) or "")[:120],
        "backendDOMNodeId": node.get("backendDOMNodeId"),
        "box": box,
    }


def collect_rows_for_prefix(sample_dir, prefix, args, rows, stats):
    screenshot = sample_dir / f"{prefix}-screenshot.webp"
    axtree_path = sample_dir / f"{prefix}-axtree.json.gz"
    bb_path = sample_dir / f"{prefix}-bb.json.gz"
    if not screenshot.exists() or not axtree_path.exists() or not bb_path.exists():
        stats["missing_prefix_files"] += 1
        return

    with Image.open(screenshot) as image:
        image_w, image_h = image.size

    axtree = load_gzip_json(axtree_path)
    boxes = load_gzip_json(bb_path)
    nodes = axtree.get("nodes", [])
    nodes_by_id = {str(node.get("nodeId")): node for node in nodes if isinstance(node, dict)}

    for node in nodes:
        if not isinstance(node, dict) or node.get("ignored"):
            continue

        parent_box = valid_box(node, boxes, image_w, image_h)
        if not parent_box or box_area(parent_box) < args.min_parent_area:
            continue

        normalized_parent, children, parent_collapsed, child_collapsed = effective_children(
            node, nodes_by_id, boxes, image_w, image_h, args.min_child_area
        )
        normalized_box = valid_box(normalized_parent, boxes, image_w, image_h)
        if not normalized_box:
            continue
        if (
            normalized_box[2] > args.max_crop_width
            or normalized_box[3] > args.max_crop_height
            or box_area(normalized_box) > args.max_crop_area
        ):
            continue

        child_boxes = []
        child_meta = []
        for child in children:
            child_box = valid_box(child, boxes, image_w, image_h)
            rel = relative_box(child_box, normalized_box) if child_box else None
            if rel and box_area(rel) >= args.min_child_area:
                child_boxes.append(rel)
                child_meta.append(payload(child, rel))

        if len(child_boxes) < args.min_children:
            continue

        rows.append({
            "site": sample_dir.name,
            "webui_sample": sample_dir.name,
            "webui_prefix": prefix,
            "image_path": str(screenshot.resolve()),
            "crop_box": normalized_box,
            "crop_size": [normalized_box[2], normalized_box[3]],
            "image_size": [image_w, image_h],
            "source_viewport": prefix,
            "box_scale": [1.0, 1.0],
            "child_boxes": child_boxes,
            "child_count": len(child_boxes),
            "parent": payload(normalized_parent, normalized_box),
            "children": child_meta,
            "depth": 0,
            "collapsed_parent_levels": parent_collapsed,
            "collapsed_child_levels": child_collapsed,
            "source_dataset": "webui-7kbal",
        })
        stats["rows"] += 1
        stats["child_boxes"] += len(child_boxes)
        stats[f"prefix:{prefix}"] += 1


def sample_prefixes(sample_dir, device):
    paths = sorted(sample_dir.glob("*-screenshot.webp"))
    prefixes = [p.name[: -len("-screenshot.webp")] for p in paths]
    if device == "all":
        return prefixes
    return [prefix for prefix in prefixes if prefix == device]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows, eval_fraction, seed):
    by_sample = defaultdict(list)
    for row in rows:
        by_sample[row["webui_sample"]].append(row)

    sample_ids = sorted(by_sample)
    random.Random(seed).shuffle(sample_ids)
    eval_count = max(1, round(len(sample_ids) * eval_fraction)) if sample_ids else 0
    eval_ids = set(sample_ids[:eval_count])

    train_rows = []
    eval_rows = []
    for sample_id in sample_ids:
        target = eval_rows if sample_id in eval_ids else train_rows
        target.extend(by_sample[sample_id])
    return train_rows, eval_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("webui_dir", type=Path, help="Directory containing WebUI sample folders")
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("eval_jsonl", type=Path)
    parser.add_argument("--device", default="default_1280-720", help="Device prefix, or 'all'")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--eval_fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_children", type=int, default=2)
    parser.add_argument("--min_parent_area", type=int, default=512)
    parser.add_argument("--min_child_area", type=int, default=64)
    parser.add_argument("--max_crop_width", type=int, default=3000)
    parser.add_argument("--max_crop_height", type=int, default=1800)
    parser.add_argument("--max_crop_area", type=int, default=4_500_000)
    args = parser.parse_args()

    sample_dirs = sorted([p for p in args.webui_dir.iterdir() if p.is_dir()])
    if args.max_samples:
        sample_dirs = sample_dirs[: args.max_samples]

    rows = []
    stats = Counter()
    for index, sample_dir in enumerate(sample_dirs, 1):
        for prefix in sample_prefixes(sample_dir, args.device):
            collect_rows_for_prefix(sample_dir, prefix, args, rows, stats)
        if index % 500 == 0:
            print(f"Scanned {index}/{len(sample_dirs)} samples; rows={len(rows)}")

    train_rows, eval_rows = split_rows(rows, args.eval_fraction, args.seed)
    write_jsonl(args.train_jsonl, train_rows)
    write_jsonl(args.eval_jsonl, eval_rows)

    print(f"Samples scanned: {len(sample_dirs)}")
    print(f"Rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Eval rows: {len(eval_rows)}")
    print(f"Child boxes: {stats['child_boxes']}")
    print("Top prefixes:", ", ".join(f"{k[7:]}={v}" for k, v in stats.most_common() if k.startswith("prefix:"))[:240])


if __name__ == "__main__":
    main()
