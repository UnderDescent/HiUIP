# train_sam2_decoder.py
#
# Fine-tunes only SAM2's mask decoder (vision encoder + prompt encoder stay
# frozen) on (image, point, target_box) pairs from build_training_pairs.py.
#
# Usage:
#   python train_sam2_decoder.py \
#       --train train_pairs.jsonl --eval eval_pairs.jsonl \
#       --epochs 1 --max_pairs 50 --output_dir checkpoint
#
# Start with --max_pairs 20-50 on your Mac as a dry run to confirm nothing
# crashes before running the full thing on the 4090 (drop --max_pairs there).
#
# API used here is taken directly from HuggingFace's documented Sam2Model /
# Sam2Processor reference (huggingface.co/docs/transformers/main/en/model_doc/sam2)
# as of July 2026 -- field names (pixel_values, input_points, input_labels,
# pred_masks, iou_scores) and shapes are from that doc, not guessed.

import argparse
import json
import random
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam2Model, Sam2Processor


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pairs(path, max_pairs=None):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    random.seed(42)
    random.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def box_to_mask(box, img_w, img_h):
    """Rasterize an [x,y,w,h] box into a full-resolution binary mask."""
    x, y, w, h = box
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)


def freeze_encoder_params(model):
    """Freeze vision encoder + prompt encoder, leave mask decoder trainable.
    This is the standard SAM fine-tuning recipe -- the decoder is small
    (~4M params) and is what actually needs to adapt to a new domain; the
    encoder's general visual features don't need to change."""
    trainable, frozen = 0, 0
    for name, param in model.named_parameters():
        if "vision_encoder" in name or "prompt_encoder" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    print(f"Frozen params: {frozen:,} | Trainable params: {trainable:,}")


def forward_one(model, processor, image, point_xy, device):
    """Runs one (image, point) through the model. Returns outputs + inputs
    (inputs needed later for post_process_masks, which requires original_sizes)."""
    input_points = [[[[point_xy[0], point_xy[1]]]]]  # image, object, point, coord
    input_labels = [[[1]]]  # positive click

    inputs = processor(
        images=image,
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
    # pred_masks shape: (batch=1, point_batch=1, num_masks=1, h, w) -- low-res logits
    pred_logits = outputs.pred_masks.squeeze(0).squeeze(0).squeeze(0)  # -> (h, w)

    full_res_mask = box_to_mask(target_box, img_w, img_h).to(device)
    # Resize target to whatever resolution the model actually predicted at --
    # avoids hardcoding a resolution number that might be wrong.
    target_resized = F.interpolate(
        full_res_mask[None, None, :, :], size=pred_logits.shape, mode="nearest"
    ).squeeze()

    bce = F.binary_cross_entropy_with_logits(pred_logits, target_resized)
    dice = dice_loss(pred_logits, target_resized)
    return bce + dice


def compute_iou(outputs, inputs, processor, target_box, img_w, img_h):
    """IoU between the model's predicted mask (post-processed to full image
    size) and the ground-truth box, for eval reporting."""
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), inputs["original_sizes"].cpu()
    )[0]
    pred_mask = masks.squeeze().bool()

    target_mask = box_to_mask(target_box, img_w, img_h).bool()
    # Align shapes defensively in case post-processing size differs slightly
    if pred_mask.shape != target_mask.shape:
        target_mask = F.interpolate(
            target_mask[None, None].float(), size=pred_mask.shape, mode="nearest"
        ).squeeze().bool()

    intersection = (pred_mask & target_mask).sum().item()
    union = (pred_mask | target_mask).sum().item()
    return intersection / union if union > 0 else 0.0


def evaluate(model, processor, pairs, device, label):
    model.eval()
    ious = []
    with torch.no_grad():
        for i, pair in enumerate(pairs):
            image = Image.open(pair["image_path"]).convert("RGB")
            outputs, inputs = forward_one(model, processor, image, pair["point"], device)
            iou = compute_iou(outputs, inputs, processor, pair["target_box"], image.width, image.height)
            ious.append(iou)
            if i < 3:
                print(f"  [{label} sample {i}] IoU: {iou:.3f}  tag={pair.get('tag')}")
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"{label} mean IoU over {len(ious)} examples: {mean_iou:.4f}")
    return mean_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model_name", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_pairs", type=int, default=None, help="cap for quick dry runs")
    parser.add_argument("--output_dir", default="sam2_finetuned")
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading {args.model_name}...")
    model = Sam2Model.from_pretrained(args.model_name).to(device)
    processor = Sam2Processor.from_pretrained(args.model_name)

    freeze_encoder_params(model)

    train_pairs = load_pairs(args.train, args.max_pairs)
    eval_pairs = load_pairs(args.eval, args.max_pairs)
    print(f"Train pairs: {len(train_pairs)} | Eval pairs: {len(eval_pairs)}")

    print("\n=== Zero-shot baseline (before any training) ===")
    baseline_iou = evaluate(model, processor, eval_pairs, device, "baseline")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    print("\n=== Training ===")
    model.train()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(train_pairs)
        running_loss = 0.0
        for pair in train_pairs:
            image = Image.open(pair["image_path"]).convert("RGB")
            outputs, inputs = forward_one(model, processor, image, pair["point"], device)
            loss = compute_loss(outputs, pair["target_box"], image.width, image.height, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}: loss {running_loss / args.log_every:.4f}")
                running_loss = 0.0

    print("\n=== Fine-tuned evaluation ===")
    finetuned_iou = evaluate(model, processor, eval_pairs, device, "fine-tuned")

    print(f"\nBaseline mean IoU:   {baseline_iou:.4f}")
    print(f"Fine-tuned mean IoU: {finetuned_iou:.4f}")
    print(f"Delta:               {finetuned_iou - baseline_iou:+.4f}")

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nSaved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
