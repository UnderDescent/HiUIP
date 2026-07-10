#!/usr/bin/env python3
"""Fine-tune SAM2 on direct-child UI objects.

Training target:
  image  = parent DOM element cropped from the screenshot
  prompt = one positive point inside one direct child
  mask   = that direct child's full box within the parent crop

This teaches SAM2 that nested text/icons inside a direct child belong to the
same object, while the rest of the parent crop is background for that prompt.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import Sam2Model, Sam2Processor


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_direct_child_samples(path, max_samples=None):
    samples = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for child_index, child_box in enumerate(row["child_boxes"]):
                samples.append({
                    "image_path": row["image_path"],
                    "crop_box": row["crop_box"],
                    "target_box": child_box,
                    "all_child_boxes": row["child_boxes"],
                    "site": row.get("site"),
                    "parent_tag": row.get("parent", {}).get("tag"),
                    "child_tag": row.get("children", [{}])[child_index].get("tag")
                    if child_index < len(row.get("children", []))
                    else None,
                })

    random.seed(42)
    random.shuffle(samples)
    return samples[:max_samples] if max_samples else samples


def crop_parent_image(sample):
    image = Image.open(sample["image_path"]).convert("RGB")
    x, y, w, h = sample["crop_box"]
    return image.crop((x, y, x + w, y + h))


def sample_point_inside_box(box):
    x, y, w, h = box
    px = x + random.uniform(0.2, 0.8) * w
    py = y + random.uniform(0.2, 0.8) * h
    return [round(px, 1), round(py, 1)]


def point_in_box(point, box):
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def sample_background_point(img_w, img_h, child_boxes, max_attempts=100):
    for _ in range(max_attempts):
        point = [random.uniform(0, img_w - 1), random.uniform(0, img_h - 1)]
        if not any(point_in_box(point, box) for box in child_boxes):
            return [round(point[0], 1), round(point[1], 1)]
    return None


def build_prompt(sample, img_w, img_h, negative_siblings, negative_background):
    points = [sample_point_inside_box(sample["target_box"])]
    labels = [1]

    sibling_boxes = [box for box in sample.get("all_child_boxes", []) if box != sample["target_box"]]
    random.shuffle(sibling_boxes)
    for box in sibling_boxes[:negative_siblings]:
        points.append(sample_point_inside_box(box))
        labels.append(0)

    for _ in range(negative_background):
        point = sample_background_point(img_w, img_h, sample.get("all_child_boxes", []))
        if point is not None:
            points.append(point)
            labels.append(0)

    return points, labels


def box_to_mask(box, img_w, img_h):
    x, y, w, h = box
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)


def freeze_encoder_params(model):
    trainable, frozen = 0, 0
    for name, param in model.named_parameters():
        if "vision_encoder" in name or "prompt_encoder" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    print(f"Frozen params: {frozen:,} | Trainable params: {trainable:,}")


def forward_one(model, processor, image, points, labels, device):
    input_points = [[points]]
    input_labels = [[labels]]
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)
    outputs = model(**inputs, multimask_output=False)
    return outputs, inputs


def forward_batch(model, processor, batch_items, device):
    images = [item["image"] for item in batch_items]
    input_points = [[item["points"]] for item in batch_items]
    input_labels = [[item["labels"]] for item in batch_items]
    inputs = processor(
        images=images,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)
    outputs = model(**inputs, multimask_output=False)
    return outputs, inputs


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + eps) / (union + eps)


def compute_loss(outputs, target_box, img_w, img_h, device):
    pred_logits = outputs.pred_masks.squeeze(0).squeeze(0).squeeze(0)
    full_res_mask = box_to_mask(target_box, img_w, img_h).to(device)
    target_resized = F.interpolate(
        full_res_mask[None, None, :, :],
        size=pred_logits.shape,
        mode="nearest",
    ).squeeze()
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_resized)
    return bce + dice_loss(pred_logits, target_resized)


def compute_batch_loss(outputs, batch_items, device):
    pred_logits = outputs.pred_masks.squeeze(2).squeeze(1)
    losses = []
    for index, item in enumerate(batch_items):
        full_res_mask = box_to_mask(
            item["sample"]["target_box"],
            item["image"].width,
            item["image"].height,
        ).to(device)
        target_resized = F.interpolate(
            full_res_mask[None, None, :, :],
            size=pred_logits.shape[-2:],
            mode="nearest",
        ).squeeze()
        bce = F.binary_cross_entropy_with_logits(pred_logits[index], target_resized)
        losses.append(bce + dice_loss(pred_logits[index], target_resized))
    return torch.stack(losses).mean()


def compute_iou(outputs, inputs, processor, target_box, img_w, img_h):
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].cpu(),
    )[0]
    pred_mask = masks.squeeze().bool()
    target_mask = box_to_mask(target_box, img_w, img_h).bool()
    if pred_mask.shape != target_mask.shape:
        target_mask = F.interpolate(
            target_mask[None, None].float(),
            size=pred_mask.shape,
            mode="nearest",
        ).squeeze().bool()

    intersection = (pred_mask & target_mask).sum().item()
    union = (pred_mask | target_mask).sum().item()
    return intersection / union if union > 0 else 0.0


def evaluate(model, processor, samples, device, label):
    model.eval()
    ious = []
    random.seed(123)
    with torch.no_grad():
        for index, sample in enumerate(samples):
            image = crop_parent_image(sample)
            point = sample_point_inside_box(sample["target_box"])
            outputs, inputs = forward_one(model, processor, image, [point], [1], device)
            iou = compute_iou(
                outputs,
                inputs,
                processor,
                sample["target_box"],
                image.width,
                image.height,
            )
            ious.append(iou)
            if index < 5:
                print(
                    f"  [{label} sample {index}] IoU={iou:.3f} "
                    f"parent={sample.get('parent_tag')} child={sample.get('child_tag')} "
                    f"site={sample.get('site')}"
                )
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"{label} mean IoU over {len(ious)} child objects: {mean_iou:.4f}")
    return mean_iou


def make_batches(samples, batch_size):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model_name", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="sam2_direct_children_finetuned")
    parser.add_argument("--cache_dir", default="data_import/hf_cache")
    parser.add_argument("--negative_siblings", type=int, default=0)
    parser.add_argument("--negative_background", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every_steps", type=int, default=0)
    args = parser.parse_args()

    random.seed(42)
    device = get_device()
    print(f"Device: {device}")

    print(f"Loading {args.model_name}...")
    model = Sam2Model.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(device)
    processor = Sam2Processor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    freeze_encoder_params(model)

    max_train_samples = args.max_train_samples if args.max_train_samples is not None else args.max_samples
    max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else args.max_samples
    train_samples = load_direct_child_samples(args.train, max_train_samples)
    eval_samples = load_direct_child_samples(args.eval, max_eval_samples)
    print(f"Train child objects: {len(train_samples)} | Eval child objects: {len(eval_samples)}")

    print("\n=== Zero-shot baseline ===")
    baseline_iou = evaluate(model, processor, eval_samples, device, "baseline")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    print("\n=== Training ===")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_samples)
        running_loss = 0.0
        running_items = 0
        epoch_start_time = time.time()
        next_save_step = args.save_every_steps if args.save_every_steps else None
        for batch_samples in make_batches(train_samples, max(1, args.batch_size)):
            batch_items = []
            for sample in batch_samples:
                image = crop_parent_image(sample)
                points, labels = build_prompt(
                    sample,
                    image.width,
                    image.height,
                    args.negative_siblings,
                    args.negative_background,
                )
                batch_items.append({
                    "sample": sample,
                    "image": image,
                    "points": points,
                    "labels": labels,
                })

            if len(batch_items) == 1:
                item = batch_items[0]
                outputs, _ = forward_one(
                    model,
                    processor,
                    item["image"],
                    item["points"],
                    item["labels"],
                    device,
                )
                loss = compute_loss(
                    outputs,
                    item["sample"]["target_box"],
                    item["image"].width,
                    item["image"].height,
                    device,
                )
            else:
                outputs, _ = forward_batch(model, processor, batch_items, device)
                loss = compute_batch_loss(outputs, batch_items, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_items)
            running_items += len(batch_items)
            step += len(batch_items)
            if step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - epoch_start_time)
                seen = min(step, len(train_samples))
                rate = seen / elapsed
                remaining = max(0, len(train_samples) - seen)
                eta_minutes = remaining / max(1e-6, rate) / 60
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"train {seen}/{len(train_samples)} ({100 * seen / max(1, len(train_samples)):.1f}%) "
                    f"loss={running_loss / max(1, running_items):.4f} "
                    f"speed={rate:.1f} obj/s eta={eta_minutes:.1f}m",
                    flush=True,
                )
                running_loss = 0.0
                running_items = 0
            if next_save_step is not None and step >= next_save_step:
                checkpoint_dir = Path(f"{args.output_dir}_step{next_save_step}")
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                print(f"Saved intermediate checkpoint to {checkpoint_dir}", flush=True)
                next_save_step += args.save_every_steps

    print("\n=== Fine-tuned evaluation ===")
    finetuned_iou = evaluate(model, processor, eval_samples, device, "fine-tuned")
    print(f"\nBaseline mean IoU:   {baseline_iou:.4f}")
    print(f"Fine-tuned mean IoU: {finetuned_iou:.4f}")
    print(f"Delta:               {finetuned_iou - baseline_iou:+.4f}")

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nSaved fine-tuned SAM2 model to {args.output_dir}")


if __name__ == "__main__":
    main()
