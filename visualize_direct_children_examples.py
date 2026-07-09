#!/usr/bin/env python3
"""Render random direct-child dataset examples as annotated crop images."""

import argparse
import html
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    "#ff3b30", "#007aff", "#34c759", "#ff9500", "#af52de", "#00c7be",
    "#ff2d55", "#5856d6", "#64d2ff", "#ffd60a", "#30d158", "#bf5af2",
]


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def draw_label(draw, box, text, color):
    x, y, w, h = box
    label = f" {text} "
    bbox = draw.textbbox((x, y), label)
    label_w = bbox[2] - bbox[0]
    label_h = bbox[3] - bbox[1]
    label_y = y if y + label_h + 4 < y + h else max(0, y - label_h - 4)
    draw.rectangle([x, label_y, x + label_w + 2, label_y + label_h + 3], fill=color)
    draw.text((x + 1, label_y + 1), label, fill="white")


def render_example(row, output_path):
    image = Image.open(row["image_path"]).convert("RGB")
    x, y, w, h = row["crop_box"]
    crop = image.crop((x, y, x + w, y + h))
    draw = ImageDraw.Draw(crop)

    try:
        ImageFont.load_default()
    except OSError:
        pass

    for index, child in enumerate(row["children"]):
        box = child["box"]
        cx, cy, cw, ch = box
        color = COLORS[index % len(COLORS)]
        draw.rectangle([cx, cy, cx + cw, cy + ch], outline=color, width=3)
        label_bits = [str(index + 1), child.get("tag") or "?"]
        if child.get("role"):
            label_bits.append(child["role"])
        draw_label(draw, box, " ".join(label_bits), color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    random.seed(args.seed)
    sample = random.sample(rows, min(args.count, len(rows)))

    cards = []
    for i, row in enumerate(sample, 1):
        filename = f"example_{i:02d}.png"
        render_example(row, args.output_dir / filename)
        parent = row["parent"]
        title = f"{row['site']} | parent {parent.get('tag') or '?'} | children {row['child_count']}"
        cards.append((filename, title, row))

    html_path = args.output_dir / "index.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write("""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Direct Child Dataset Examples</title>
  <style>
    body { font: 14px system-ui, sans-serif; margin: 24px; color: #1f2328; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    img { display: block; max-width: 100%; height: auto; border: 1px solid #d8dee4; }
    h2 { font-size: 15px; margin: 0 0 8px; }
    p { margin: 8px 0 0; color: #57606a; }
  </style>
</head>
<body>
<h1>Direct Child Dataset Examples</h1>
<div class="grid">
""")
        for filename, title, row in cards:
            f.write("<section class=\"card\">\n")
            f.write(f"<h2>{html.escape(title)}</h2>\n")
            f.write(f"<img src=\"{html.escape(filename)}\" alt=\"{html.escape(title)}\">\n")
            crop = row["crop_box"]
            f.write(
                f"<p>crop: {crop}; collapsed parent levels: "
                f"{row['collapsed_parent_levels']}; collapsed child levels: "
                f"{row['collapsed_child_levels']}</p>\n"
            )
            f.write("</section>\n")
        f.write("</div>\n</body>\n</html>\n")

    print(f"Rendered {len(sample)} examples to {args.output_dir}")
    print(f"Open {html_path}")


if __name__ == "__main__":
    main()
