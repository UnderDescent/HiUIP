#!/usr/bin/env python3
"""Visualize SAM2 direct-child eval results side by side.

For each direct child in the eval JSONL:
  left  = parent crop with ground-truth child box
  right = parent crop with SAM2 predicted mask/box from a point prompt
"""

import argparse
import html
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def load_samples(path, limit=None, seed=42):
    samples = []
    with Path(path).open(encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            for child_index, child_box in enumerate(row["child_boxes"]):
                child_meta = row.get("children", [{}])[child_index]
                samples.append({
                    "row_index": row_index,
                    "child_index": child_index,
                    "image_path": row["image_path"],
                    "crop_box": row["crop_box"],
                    "target_box": child_box,
                    "site": row.get("site"),
                    "parent_tag": row.get("parent", {}).get("tag"),
                    "child_tag": child_meta.get("tag"),
                    "child_role": child_meta.get("role"),
                })
    if limit:
        random.seed(seed)
        random.shuffle(samples)
        samples = samples[:limit]
    return samples


def crop_parent_image(sample):
    image = Image.open(sample["image_path"]).convert("RGB")
    x, y, w, h = sample["crop_box"]
    return image.crop((x, y, x + w, y + h))


def point_for_box(box, rng):
    x, y, w, h = box
    px = x + rng.uniform(0.2, 0.8) * w
    py = y + rng.uniform(0.2, 0.8) * h
    return [round(px, 1), round(py, 1)]


def box_to_mask(box, img_w, img_h):
    x, y, w, h = box
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)


def mask_to_box(mask):
    ys, xs = torch.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min().item())
    y1 = int(ys.min().item())
    x2 = int(xs.max().item())
    y2 = int(ys.max().item())
    return [x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)]


def box_iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def mask_iou(pred_mask, target_box, img_w, img_h):
    target = box_to_mask(target_box, img_w, img_h).bool()
    if pred_mask.shape != target.shape:
        target = F.interpolate(
            target[None, None].float(),
            size=pred_mask.shape,
            mode="nearest",
        ).squeeze().bool()
    intersection = (pred_mask & target).sum().item()
    union = (pred_mask | target).sum().item()
    return intersection / union if union else 0.0


def forward_one(model, processor, image, point_xy, device):
    inputs = processor(
        images=image,
        input_points=[[[[point_xy[0], point_xy[1]]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].cpu(),
    )[0]
    mask = masks.squeeze()
    if mask.dtype != torch.bool:
        mask = mask > 0
    return mask


def draw_box(draw, box, color, label):
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    text = f" {label} "
    tb = draw.textbbox((x, y), text)
    label_h = tb[3] - tb[1]
    label_w = tb[2] - tb[0]
    y0 = y if y + label_h + 4 < y + h else max(0, y - label_h - 4)
    draw.rectangle([x, y0, x + label_w + 2, y0 + label_h + 3], fill=color)
    draw.text((x + 1, y0 + 1), text, fill="white")


def draw_point(draw, point):
    x, y = point
    r = 5
    draw.ellipse([x - r, y - r, x + r, y + r], fill=POINT_COLOR, outline="white", width=2)


def overlay_mask(image, mask, color=(255, 59, 48), alpha=90):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask.numpy().astype(np.uint8) * alpha), mode="L")
    fill = Image.new("RGBA", image.size, (*color, alpha))
    overlay.paste(fill, mask=mask_img)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_result(sample, model, processor, device, gt_output_path, pred_output_path, rng):
    crop = crop_parent_image(sample)
    point = point_for_box(sample["target_box"], rng)
    pred_mask = forward_one(model, processor, crop, point, device)
    pred_box = mask_to_box(pred_mask)

    gt_panel = crop.copy()
    gt_draw = ImageDraw.Draw(gt_panel)
    draw_box(gt_draw, sample["target_box"], GT_COLOR, "gt")
    draw_point(gt_draw, point)

    pred_panel = overlay_mask(crop, pred_mask)
    pred_draw = ImageDraw.Draw(pred_panel)
    if pred_box:
        draw_box(pred_draw, pred_box, PRED_COLOR, "pred")
    draw_point(pred_draw, point)

    gt_panel.save(gt_output_path)
    pred_panel.save(pred_output_path)

    iou = mask_iou(pred_mask, sample["target_box"], crop.width, crop.height)
    box_iou = box_iou_xywh(pred_box, sample["target_box"]) if pred_box else 0.0
    return {
        "mask_iou": iou,
        "box_iou": box_iou,
        "pred_box": pred_box,
        "point": point,
        "crop_size": [crop.width, crop.height],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval", default="data_import/eval_direct_children.jsonl")
    parser.add_argument("--output_dir", default="data_import/sam2_eval_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading {args.checkpoint}...")
    model = Sam2Model.from_pretrained(args.checkpoint).to(device)
    processor = Sam2Processor.from_pretrained(args.checkpoint)
    model.eval()

    samples = load_samples(args.eval, args.limit, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = random.Random(args.seed)
    for index, sample in enumerate(samples, 1):
        gt_filename = f"result_{index:03d}_gt.png"
        pred_filename = f"result_{index:03d}_pred.png"
        metrics = render_result(
            sample,
            model,
            processor,
            device,
            output_dir / gt_filename,
            output_dir / pred_filename,
            rng,
        )
        row = {**sample, **metrics, "gt_filename": gt_filename, "pred_filename": pred_filename}
        rows.append(row)
        print(
            f"{index:03d}/{len(samples)} IoU={metrics['mask_iou']:.3f} "
            f"site={sample['site']} parent={sample['parent_tag']} child={sample['child_tag']}"
        )

    mean_iou = sum(r["mask_iou"] for r in rows) / len(rows) if rows else 0.0
    mean_box_iou = sum(r["box_iou"] for r in rows) / len(rows) if rows else 0.0

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"mean_mask_iou": mean_iou, "mean_box_iou": mean_box_iou, "results": rows},
            f,
            indent=2,
        )

    with (output_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SAM2 Direct-Child Eval Results</title>
  <style>
    body { font: 14px system-ui, sans-serif; margin: 24px; color: #1f2328; }
    .summary { margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(620px, 1fr)); gap: 18px; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
    .panel-title { font-size: 12px; font-weight: 700; margin: 0 0 6px; color: #57606a; text-transform: uppercase; }
    img { display: block; max-width: 100%; height: auto; border: 1px solid #d8dee4; }
    h2 { font-size: 15px; margin: 0 0 8px; }
    p { margin: 8px 0 0; color: #57606a; }
  </style>
</head>
<body>
""")
        f.write("<h1>SAM2 Direct-Child Eval Results</h1>\n")
        f.write(
            f"<p class=\"summary\">Left: ground truth only. Right: prediction only. "
            f"Mean mask IoU: {mean_iou:.4f}; mean box IoU: {mean_box_iou:.4f}; "
            f"examples: {len(rows)}.</p>\n"
        )
        f.write("<div class=\"grid\">\n")
        for row in rows:
            title = (
                f"{row['site']} | parent {row.get('parent_tag') or '?'} | "
                f"child {row.get('child_tag') or '?'}"
            )
            f.write("<section class=\"card\">\n")
            f.write(f"<h2>{html.escape(title)}</h2>\n")
            f.write("<div class=\"pair\">\n")
            f.write("<div><p class=\"panel-title\">Ground truth</p>\n")
            f.write(f"<img src=\"{html.escape(row['gt_filename'])}\" alt=\"{html.escape(title)} ground truth\"></div>\n")
            f.write("<div><p class=\"panel-title\">Prediction</p>\n")
            f.write(f"<img src=\"{html.escape(row['pred_filename'])}\" alt=\"{html.escape(title)} prediction\"></div>\n")
            f.write("</div>\n")
            f.write(
                f"<p>mask IoU: {row['mask_iou']:.3f}; box IoU: {row['box_iou']:.3f}; "
                f"crop: {row['crop_size']}; point: {row['point']}</p>\n"
            )
            f.write("</section>\n")
        f.write("</div>\n</body>\n</html>\n")

    print(f"\nMean mask IoU: {mean_iou:.4f}")
    print(f"Mean box IoU:  {mean_box_iou:.4f}")
    print(f"Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
