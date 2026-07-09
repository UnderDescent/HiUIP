#!/usr/bin/env python3
"""Run SAM2 with a grid of point prompts over eval parent crops.

This approximates SAM-style automatic mask generation:
  1. crop to the parent element
  2. prompt SAM2 at grid points
  3. convert masks to boxes
  4. filter low-quality / tiny / duplicate masks
  5. visualize predicted child boxes against all ground-truth direct children
"""

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam2Model, Sam2Processor


GT_COLOR = "#34c759"
PRED_COLOR = "#ff3b30"
POINT_COLOR = "#007aff"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_rows(path, limit=None):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def crop_parent(row):
    image = Image.open(row["image_path"]).convert("RGB")
    x, y, w, h = row["crop_box"]
    return image.crop((x, y, x + w, y + h))


def make_grid(width, height, stride, edge_margin, min_points_per_axis, max_points_per_axis):
    # Fixed stride already gives larger crops more points, but very small crops
    # need a minimum number of probes so narrow nav/list items are not skipped.
    x_count = max(min_points_per_axis, math.ceil(width / stride))
    y_count = max(min_points_per_axis, math.ceil(height / stride))
    x_count = min(max_points_per_axis, x_count)
    y_count = min(max_points_per_axis, y_count)

    if x_count == 1:
        xs = [width / 2]
    else:
        start = min(edge_margin, max(0, width - 1))
        end = max(start, width - 1 - edge_margin)
        xs = [start + (end - start) * i / (x_count - 1) for i in range(x_count)]

    if y_count == 1:
        ys = [height / 2]
    else:
        start = min(edge_margin, max(0, height - 1))
        end = max(start, height - 1 - edge_margin)
        ys = [start + (end - start) * i / (y_count - 1) for i in range(y_count)]

    return [[round(x, 1), round(y, 1)] for y in ys for x in xs]


def make_grid_fixed(width, height, stride, edge_margin):
    points = []
    x0 = min(edge_margin, max(0, width - 1))
    y0 = min(edge_margin, max(0, height - 1))
    xs = list(range(x0, max(x0 + 1, width - edge_margin), stride))
    ys = list(range(y0, max(y0 + 1, height - edge_margin), stride))
    if not xs or xs[-1] < width - edge_margin - 1:
        xs.append(max(0, width - edge_margin - 1))
    if not ys or ys[-1] < height - edge_margin - 1:
        ys.append(max(0, height - edge_margin - 1))
    for y in ys:
        for x in xs:
            points.append([float(x), float(y)])
    return points


def mask_to_box(mask):
    ys, xs = torch.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min().item())
    y1 = int(ys.min().item())
    x2 = int(xs.max().item())
    y2 = int(ys.max().item())
    return [x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)]


def area(box):
    return box[2] * box[3]


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def best_gt_iou(pred_box, gt_boxes):
    return max((box_iou(pred_box, gt) for gt in gt_boxes), default=0.0)


def recall_at_iou(pred_boxes, gt_boxes, threshold):
    if not gt_boxes:
        return 0.0
    hit = 0
    for gt in gt_boxes:
        if max((box_iou(pred, gt) for pred in pred_boxes), default=0.0) >= threshold:
            hit += 1
    return hit / len(gt_boxes)


def prompt_batch(model, processor, image, points, device):
    if not points:
        return []
    input_points = [[[point] for point in points]]
    input_labels = [[[1] for _ in points]]
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].cpu(),
    )[0]
    masks = masks.squeeze(1) if masks.ndim == 4 else masks
    scores = outputs.iou_scores.detach().cpu().reshape(-1).tolist()
    results = []
    for point, mask, score in zip(points, masks, scores):
        if mask.dtype != torch.bool:
            mask = mask > 0
        box = mask_to_box(mask)
        if box:
            results.append({"point": point, "box": box, "score": float(score), "mask": mask})
    return results


def nms(candidates, iou_threshold):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if all(box_iou(candidate["box"], existing["box"]) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def run_grid(model, processor, image, device, args):
    if args.fixed_stride:
        points = make_grid_fixed(image.width, image.height, args.grid_stride, args.edge_margin)
    else:
        points = make_grid(
            image.width,
            image.height,
            args.grid_stride,
            args.edge_margin,
            args.min_points_per_axis,
            args.max_points_per_axis,
        )
    candidates = []
    for start in range(0, len(points), args.batch_points):
        candidates.extend(
            prompt_batch(model, processor, image, points[start:start + args.batch_points], device)
        )

    image_area = image.width * image.height
    filtered = []
    for candidate in candidates:
        box = candidate["box"]
        mask_area = int(candidate["mask"].sum().item())
        candidate["mask_area"] = mask_area
        if candidate["score"] < args.score_threshold:
            continue
        if area(box) < args.min_box_area or mask_area < args.min_mask_area:
            continue
        if area(box) / image_area > args.max_box_area_fraction:
            continue
        filtered.append(candidate)
    return nms(filtered, args.nms_iou), len(points), len(candidates)


def draw_box(draw, box, color, label):
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    text = f" {label} "
    tb = draw.textbbox((x, y), text)
    label_w = tb[2] - tb[0]
    label_h = tb[3] - tb[1]
    y0 = y if y + label_h + 4 < y + h else max(0, y - label_h - 4)
    draw.rectangle([x, y0, x + label_w + 2, y0 + label_h + 3], fill=color)
    draw.text((x + 1, y0 + 1), text, fill="white")


def draw_point(draw, point):
    x, y = point
    r = 3
    draw.ellipse([x - r, y - r, x + r, y + r], fill=POINT_COLOR)


def render_row(row, predictions, output_path):
    crop = crop_parent(row)
    gt_panel = crop.copy()
    pred_panel = crop.copy()
    gt_draw = ImageDraw.Draw(gt_panel)
    pred_draw = ImageDraw.Draw(pred_panel)

    for index, box in enumerate(row["child_boxes"], 1):
        draw_box(gt_draw, box, GT_COLOR, f"gt {index}")
        draw_box(pred_draw, box, GT_COLOR, f"gt {index}")

    for index, pred in enumerate(predictions, 1):
        draw_box(pred_draw, pred["box"], PRED_COLOR, f"p {index} {pred['score']:.2f}")
        draw_point(pred_draw, pred["point"])

    gap = 16
    out = Image.new("RGB", (gt_panel.width + pred_panel.width + gap, max(gt_panel.height, pred_panel.height)), "white")
    out.paste(gt_panel, (0, 0))
    out.paste(pred_panel, (gt_panel.width + gap, 0))
    out.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval", default="data_import/eval_direct_children.jsonl")
    parser.add_argument("--output_dir", default="data_import/sam2_grid_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grid_stride", type=int, default=64)
    parser.add_argument("--fixed_stride", action="store_true")
    parser.add_argument("--min_points_per_axis", type=int, default=4)
    parser.add_argument("--max_points_per_axis", type=int, default=24)
    parser.add_argument("--edge_margin", type=int, default=8)
    parser.add_argument("--batch_points", type=int, default=64)
    parser.add_argument("--score_threshold", type=float, default=0.35)
    parser.add_argument("--nms_iou", type=float, default=0.75)
    parser.add_argument("--min_box_area", type=int, default=64)
    parser.add_argument("--min_mask_area", type=int, default=64)
    parser.add_argument("--max_box_area_fraction", type=float, default=0.95)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading {args.checkpoint}...")
    model = Sam2Model.from_pretrained(args.checkpoint).to(device)
    processor = Sam2Processor.from_pretrained(args.checkpoint)
    model.eval()

    rows = load_rows(args.eval, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    recall50_values = []
    recall30_values = []
    pred_counts = []
    for index, row in enumerate(rows, 1):
        crop = crop_parent(row)
        preds, point_count, raw_count = run_grid(model, processor, crop, device, args)
        pred_boxes = [pred["box"] for pred in preds]
        gt_boxes = row["child_boxes"]
        recall50 = recall_at_iou(pred_boxes, gt_boxes, 0.5)
        recall30 = recall_at_iou(pred_boxes, gt_boxes, 0.3)
        pred_counts.append(len(preds))
        recall50_values.append(recall50)
        recall30_values.append(recall30)
        for pred in preds:
            pred["best_gt_iou"] = best_gt_iou(pred["box"], gt_boxes)

        filename = f"grid_{index:03d}.png"
        render_row(row, preds, output_dir / filename)
        card = {
            "filename": filename,
            "site": row.get("site"),
            "parent": row.get("parent", {}),
            "crop_size": row.get("crop_size"),
            "gt_count": len(gt_boxes),
            "pred_count": len(preds),
            "point_count": point_count,
            "raw_count": raw_count,
            "recall50": recall50,
            "recall30": recall30,
            "predictions": [
                {
                    "box": pred["box"],
                    "score": pred["score"],
                    "point": pred["point"],
                    "best_gt_iou": pred["best_gt_iou"],
                }
                for pred in preds
            ],
        }
        cards.append(card)
        print(
            f"{index:03d}/{len(rows)} gt={len(gt_boxes)} pred={len(preds)} "
            f"recall@.5={recall50:.2f} recall@.3={recall30:.2f} site={row.get('site')}"
        )

    summary = {
        "rows": len(cards),
        "mean_recall50": sum(recall50_values) / len(recall50_values) if recall50_values else 0.0,
        "mean_recall30": sum(recall30_values) / len(recall30_values) if recall30_values else 0.0,
        "mean_pred_count": sum(pred_counts) / len(pred_counts) if pred_counts else 0.0,
        "args": vars(args),
        "cards": cards,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SAM2 Grid Direct-Child Results</title>
  <style>
    body { font: 14px system-ui, sans-serif; margin: 24px; color: #1f2328; }
    .summary { margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 18px; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    img { display: block; max-width: 100%; height: auto; border: 1px solid #d8dee4; }
    h2 { font-size: 15px; margin: 0 0 8px; }
    p { margin: 8px 0 0; color: #57606a; }
  </style>
</head>
<body>
""")
        f.write("<h1>SAM2 Grid Direct-Child Results</h1>\n")
        f.write(
            f"<p class=\"summary\">Left: all ground-truth direct children. "
            f"Right: ground truth plus grid-prompt predictions. "
            f"Mean recall@0.5: {summary['mean_recall50']:.3f}; "
            f"mean recall@0.3: {summary['mean_recall30']:.3f}; "
            f"mean predictions/crop: {summary['mean_pred_count']:.1f}.</p>\n"
        )
        f.write("<div class=\"grid\">\n")
        for card in cards:
            parent = card["parent"]
            title = f"{card['site']} | parent {parent.get('tag') or '?'}"
            f.write("<section class=\"card\">\n")
            f.write(f"<h2>{html.escape(title)}</h2>\n")
            f.write(f"<img src=\"{html.escape(card['filename'])}\" alt=\"{html.escape(title)}\">\n")
            f.write(
                f"<p>gt: {card['gt_count']}; pred: {card['pred_count']}; "
                f"grid points: {card['point_count']}; recall@0.5: {card['recall50']:.2f}; "
                f"recall@0.3: {card['recall30']:.2f}; crop: {card['crop_size']}</p>\n"
            )
            f.write("</section>\n")
        f.write("</div>\n</body>\n</html>\n")

    print(f"\nMean recall@0.5: {summary['mean_recall50']:.4f}")
    print(f"Mean recall@0.3: {summary['mean_recall30']:.4f}")
    print(f"Mean predictions/crop: {summary['mean_pred_count']:.2f}")
    print(f"Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
