#!/usr/bin/env python
# coding: utf-8
"""
Gemma3-4B-Instruct on COCO: load model, run VQA, and bbox detection.
Converted from Qwen3-VL-8B-Instruct_on_Coco.py to use Google Gemma 3 4B.
"""
# pip: torchvision transformers>=4.50.0 accelerate pillow

import json
import os
import random
import re
import shutil
import torch
import torchvision  # type: ignore[reportMissingImports]
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import zoom
from tqdm import tqdm

# Number of images to process (bbox + report + bounding_box_comparison.png only; attention disabled)
NUM_IMAGES = 200

print(f"NUM_IMAGES = {NUM_IMAGES}")
print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print("Installation successful!")


# --- Load model ---

os.environ["HF_HOME"] = "/n/netscratch/sham_lab/Lab/chloe00/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/n/netscratch/sham_lab/Lab/chloe00/huggingface"

# Use Gemma 3 4B instruction-tuned model
model_id = "google/gemma-3-4b-it"

print("Loading Gemma 3 4B model...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # Use eager for attention output support
)

processor = AutoProcessor.from_pretrained(
    model_id,
    padding_side="left"
)

print("Model loaded successfully!")
print(f"Model type: {type(model)}")

# --- Image list: use COCO clothing dataset (train.json + val.json), same as COCO_Clothing_Dataset ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_data_dir = os.path.join(_script_dir, "data")


def _load_coco_image_paths(data_dir):
    """Build list of image paths from COCO train.json + val.json (train -> data/train, val -> data/test)."""
    paths = []
    for ann_name, img_subdir in [("train.json", "train"), ("val.json", "test")]:
        ann_path = os.path.join(data_dir, ann_name)
        if not os.path.isfile(ann_path):
            continue
        with open(ann_path) as f:
            coco = json.load(f)
        images_dir = os.path.join(data_dir, img_subdir)
        for img in coco.get("images", []):
            fn = img.get("file_name")
            if not fn:
                continue
            p = os.path.join(images_dir, fn)
            if os.path.isfile(p):
                paths.append(p)
    return paths


def get_item_for_image(image_paths, filename):
    """Get path for the given image filename (e.g. fb0046f5396fc4731f4dda322433f18c.jpg)."""
    for p in image_paths:
        if os.path.basename(p) == filename or p.endswith(filename):
            return p
    return image_paths[0] if image_paths else None


_coco_paths_all = _load_coco_image_paths(_data_dir)
# Use only train split for random draw
_train_paths = [p for p in _coco_paths_all if "train" in p]
_image_paths = []
if _train_paths:
    n = min(NUM_IMAGES, len(_train_paths))
    _image_paths = random.sample(_train_paths, n)
    print(f"Image list: {len(_train_paths)} train images; randomly drawing {n}.")
if not _image_paths:
    # Fallback: scan data/train for any jpg (no COCO JSON or no existing paths)
    print("No images from COCO dataset; falling back to data/train.")
    _train_dir = os.path.join(_data_dir, "train")
    if os.path.isdir(_train_dir):
        for f in sorted(os.listdir(_train_dir)):
            if f.lower().endswith((".jpg", ".jpeg")):
                p = os.path.join(_train_dir, f)
                if os.path.isfile(p):
                    _image_paths = [p]
                    break
# Output base: always under script directory so saves are predictable
_output_base = os.path.abspath(os.path.join(_script_dir, "output_gemma3"))
os.makedirs(_output_base, exist_ok=True)
print(f"Processing {len(_image_paths)} image(s). Output folder: {_output_base}")
if not _image_paths:
    raise FileNotFoundError("No images found in COCO train dataset.")

# --- Attention map code kept but unused (focus: bounding_box_comparison.png only) ---
# NOTE: Gemma 3 attention handling differs from Qwen3-VL. The functions below are
# preserved for reference but may need adaptation for Gemma 3's SigLIP vision encoder.


def compute_attention_weights_map(model, processor, image, prompt, sharpen_power=2.0):
    """
    Attention from last token to image tokens only (no grad).
    NOTE: This function may need adaptation for Gemma 3's architecture.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )
    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise ValueError("model returned no attentions (ensure attn_implementation='eager' was used)")
    # Last layer: (batch, heads, seq_len, seq_len)
    last_attn = outputs.attentions[-1]

    # Gemma 3 uses SigLIP encoder - image token layout may differ from Qwen3-VL
    # This is a simplified approach; may need adjustment based on actual token layout
    seq_len = last_attn.shape[-1]
    # Estimate number of image patches (Gemma 3 uses 896x896 images -> 14x14 patches per crop)
    n_patches = 14 * 14  # Default SigLIP patch count

    if n_patches > seq_len:
        n_patches = seq_len // 4  # Fallback estimate

    # Attention from last token to image tokens
    attn_first = last_attn[0, :, -1, :n_patches].mean(dim=0).cpu().float().numpy()
    attn_last = last_attn[0, :, -1, -n_patches:].mean(dim=0).cpu().float().numpy()
    # Use the slice with higher variance (more informative spatial structure)
    if np.var(attn_last) > np.var(attn_first):
        attn_to_image = attn_last
    else:
        attn_to_image = attn_first

    # Reshape to 2D (assuming square grid)
    side = int(np.sqrt(len(attn_to_image)))
    if side * side == len(attn_to_image):
        importance_map = attn_to_image.reshape(side, side)
    else:
        importance_map = attn_to_image.reshape(-1, 1)  # Fallback

    # Sharpen: emphasize high-attention regions, suppress background
    importance_map = np.power(np.maximum(importance_map, 0), sharpen_power)
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
    return importance_map.astype(np.float64)


def compute_category_attention_map(model, processor, image, category_name, sharpen_power=2.0):
    """
    Category-specific attention: one forward with prompt asking only for this category.
    Returns 2D attention map (last token -> image tokens) for that category.
    """
    prompt = (
        f"Detect the {category_name} in this image. "
        "Return the bounding box coordinates as [x1, y1, x2, y2]. Only return the coordinates, nothing else."
    )
    return compute_attention_weights_map(model, processor, image, prompt, sharpen_power=sharpen_power)


def attention_bbox_metric(attention_map_2d, bbox, img_w, img_h):
    """
    Compare attention map to GT bbox: mean attention inside bbox vs outside.
    attention_map_2d: (H, W) in patch grid. bbox: (x1, y1, x2, y2) pixel coords.
    Returns dict with mean_inside, mean_outside, ratio (inside/(outside+eps)).
    """
    if attention_map_2d is None or attention_map_2d.ndim < 2:
        return None
    scale_h = img_h / attention_map_2d.shape[0]
    scale_w = img_w / attention_map_2d.shape[1]
    resized = zoom(attention_map_2d, (scale_h, scale_w), order=1)
    x1, y1, x2, y2 = bbox
    x1, x2 = int(max(0, x1)), int(min(img_w, x2))
    y1, y2 = int(max(0, y1)), int(min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    inside = resized[y1:y2, x1:x2]
    mask_out = np.ones_like(resized, dtype=bool)
    mask_out[y1:y2, x1:x2] = False
    mean_inside = float(np.mean(inside))
    mean_outside = float(np.mean(resized[mask_out])) if np.any(mask_out) else 0.0
    eps = 1e-8
    ratio = mean_inside / (mean_outside + eps)
    return {"mean_inside": mean_inside, "mean_outside": mean_outside, "ratio": ratio}


def compute_gradcam_attention(model, processor, image, question):
    """
    Compute GradCAM importance over image patches.
    NOTE: This function may need adaptation for Gemma 3's architecture.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Get pixel_values for gradient computation
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        return None
    pixel_values = pixel_values.to(model.device).requires_grad_(True)

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
    )
    logits = outputs.logits
    predicted_token = logits[0, -1].argmax()
    model.zero_grad()
    score = logits[0, -1, predicted_token]
    score.backward()

    grad = pixel_values.grad
    if grad is None:
        return None
    if grad.dim() >= 3:
        importance_per_patch = grad.abs().mean(dim=tuple(range(1, grad.dim()))).cpu().numpy()
    else:
        importance_per_patch = grad.abs().mean(dim=-1).cpu().numpy()

    # Reshape to 2D grid
    n_patches = importance_per_patch.size
    side = int(np.sqrt(n_patches))
    if side * side == n_patches:
        importance_map = importance_per_patch.reshape(side, side)
    else:
        importance_map = importance_per_patch.reshape(-1, 1)

    # Sharpen to concentrate on key regions
    importance_map = np.power(np.maximum(importance_map.astype(np.float64), 0), 2.0)
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
    return importance_map.astype(np.float64)


BBOX_COLORS = {"sleeve": "red", "shoes": "blue", "neckline": "green", "pants": "yellow"}
EXTRA_COLORS = ["orange", "purple", "cyan", "magenta", "brown", "pink", "gray", "olive"]


def _color_for_cat(cat_name):
    return BBOX_COLORS.get(cat_name) or EXTRA_COLORS[hash(cat_name) % len(EXTRA_COLORS)]


def box_iou(a, b):
    """
    IoU (Intersection over Union) for two boxes.
    a, b: (x1, y1, x2, y2). Returns float in [0, 1].
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_iou_per_category(pred_items, gt_bboxes):
    """
    IoU between predicted boxes (pred_items) and GT (gt_bboxes) per category.
    pred_items: dict category_name -> (x1,y1,x2,y2)  (one pred per category)
    gt_bboxes: dict category_name -> list of (x1,y1,x2,y2)
    Returns dict category -> iou (pred matched to best GT by IoU), and mean_iou.
    """
    gt_name_map = {"shoes": "shoe"}
    iou_per_cat = {}
    for pred_name, pred_box in pred_items.items():
        gt_name = gt_name_map.get(pred_name, pred_name)
        gt_boxes = gt_bboxes.get(gt_name, [])
        if not gt_boxes:
            iou_per_cat[pred_name] = None  # no GT
            continue
        ious = [box_iou(pred_box, gt) for gt in gt_boxes]
        iou_per_cat[pred_name] = max(ious)
    valid = [v for v in iou_per_cat.values() if v is not None]
    mean_iou = sum(valid) / len(valid) if valid else None
    return iou_per_cat, mean_iou


def load_annotation_category_names(annotation_path):
    """Return list of category names from annotation JSON (for fallback when image has no GT)."""
    if not os.path.isfile(annotation_path):
        return []
    with open(annotation_path) as f:
        data = json.load(f)
    return [c["name"] for c in data.get("categories", [])]


def load_gt_bboxes(annotation_path, image_file_name, all_categories=True):
    """
    Load ground-truth bboxes from Fashionpedia-style JSON.
    all_categories: if True, load all categories for the image; if False, only sleeve/shoe/neckline/pants.
    Returns dict: category_name -> list of (x1, y1, x2, y2). Annotation bbox is [x, y, w, h].
    """
    if not os.path.isfile(annotation_path):
        return {}
    with open(annotation_path) as f:
        data = json.load(f)
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    if not all_categories:
        target_names = {"sleeve", "shoe", "neckline", "pants"}
        target_ids = {c["id"] for c in data["categories"] if c["name"] in target_names}
    else:
        target_ids = None  # all categories
    image = next((im for im in data["images"] if im["file_name"] == image_file_name), None)
    if not image:
        return {}
    gt = {}
    for ann in data["annotations"]:
        if ann["image_id"] != image["id"]:
            continue
        if target_ids is not None and ann["category_id"] not in target_ids:
            continue
        name = id2name[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        box = (float(x), float(y), float(x + w), float(y + h))
        gt.setdefault(name, []).append(box)
    return gt


def _bbox_1000_to_pixel(x1, y1, x2, y2, img_w, img_h, letterbox_1000=True):
    """Convert bbox from model's [0,1000] space to original image pixel coords.
    If letterbox_1000, assume model uses a 1000x1000 canvas with letterboxing (content centered).
    """
    if letterbox_1000 and img_w != img_h:
        if img_w < img_h:
            cw = 1000.0 * img_w / img_h
            ox = (1000.0 - cw) / 2.0
            x1 = (x1 - ox) * img_w / cw
            x2 = (x2 - ox) * img_w / cw
            y1 = y1 * img_h / 1000.0
            y2 = y2 * img_h / 1000.0
        else:
            ch = 1000.0 * img_h / img_w
            oy = (1000.0 - ch) / 2.0
            y1 = (y1 - oy) * img_h / ch
            y2 = (y2 - oy) * img_h / ch
            x1 = x1 * img_w / 1000.0
            x2 = x2 * img_w / 1000.0
    else:
        x1, y1 = x1 * img_w / 1000.0, y1 * img_h / 1000.0
        x2, y2 = x2 * img_w / 1000.0, y2 * img_h / 1000.0
    return (x1, y1, x2, y2)


# --- Per-image: GT, bbox detection, report, figure (loop over NUM_IMAGES) ---
_data_dir = os.path.join(_script_dir, "data")

for _idx, image_path in enumerate(tqdm(_image_paths, desc="Images")):
    _img_name = os.path.basename(image_path)
    tqdm.write(f"  [{_idx + 1}/{len(_image_paths)}] {_img_name}")
    image = Image.open(image_path).convert("RGB")

    # Ground-truth bboxes from annotations (all categories for this image)
    _ann_file = "train.json" if "train" in image_path else "val.json"
    _ann_path = os.path.join(_data_dir, _ann_file)
    gt_bboxes = load_gt_bboxes(_ann_path, os.path.basename(image_path), all_categories=True)

    # Only ask VLM to detect items that have exactly 1 bbox in GT (skip e.g. shoe when shoe_1 + shoe_2)
    gt_cats_for_prompt = sorted([c for c in gt_bboxes if len(gt_bboxes[c]) == 1])
    if not gt_cats_for_prompt:
        gt_cats_for_prompt = load_annotation_category_names(_ann_path)

    # --- Run VLM bbox detection with prompt from GT labels ---
    _items_list = "\n".join(f"- {c}" for c in gt_cats_for_prompt)
    bbox_prompt = f"""Detect the following fashion items in this image and return their bounding box coordinates:
{_items_list}. The bounding boxes mean this area is highly relevant in predicting the item. For example,
if the item is a shoe, the bounding box should be the area that is highly relevant to the shoe.
Don't include the area that is not relevant to the item.

For each item found, return in this format:
item_name: [x1, y1, x2, y2]

Only return the coordinates, nothing else."""
    messages_bbox = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": bbox_prompt},
        ],
    }]
    inputs_bbox = processor.apply_chat_template(
        messages_bbox,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs_bbox = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs_bbox.items()}
    with torch.no_grad():
        generated_bbox = model.generate(**inputs_bbox, max_new_tokens=512, do_sample=False)
    output_text_bbox = processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs_bbox["input_ids"], generated_bbox)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Parse model output: match any GT category name (and "shoes" for "shoe")
    _cats_for_regex = list(gt_cats_for_prompt) + (["shoes"] if "shoe" in gt_cats_for_prompt else [])
    _cats_for_regex = sorted(set(_cats_for_regex), key=len, reverse=True)
    _pattern = r"(" + "|".join(re.escape(c) for c in _cats_for_regex) + r")[:\s]+\[?\(?([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)?\]?"

    try:
        # GT categories with exactly 1 bbox (same as VLM prompt); use for report and figure
        gt_single = {c: gt_bboxes[c] for c in gt_cats_for_prompt}

        detected_items = {}
        img_w, img_h = image.size[0], image.size[1]
        for m in re.findall(_pattern, output_text_bbox.lower()):
            name, x1, y1, x2, y2 = m[0], float(m[1]), float(m[2]), float(m[3]), float(m[4])
            max_val = max(x1, y1, x2, y2)
            if max_val <= 1.5:
                x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
            elif max_val <= 1000 and min(x1, y1, x2, y2) >= 0:
                x1, y1, x2, y2 = _bbox_1000_to_pixel(x1, y1, x2, y2, img_w, img_h, letterbox_1000=True)
            detected_items[name] = (x1, y1, x2, y2)

        def _gt_display_name(cat):
            return "shoes" if cat == "shoe" else cat

        iou_per_cat, mean_iou = compute_iou_per_category(detected_items, gt_bboxes)

        _basename = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(_output_base, _basename)
        os.makedirs(output_dir, exist_ok=True)

        def _fmt_bbox(b):
            return f"({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f})"

        lines = []
        lines.append("=" * 60)
        lines.append(f"Image: {os.path.basename(image_path)}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("GROUND-TRUTH (categories with 1 bbox only)")
        lines.append("-" * 40)
        for cat in sorted(gt_single.keys()):
            boxes = gt_single[cat]
            display = _gt_display_name(cat)
            lines.append(f"  {display}: {_fmt_bbox(boxes[0])}")
        lines.append("")
        lines.append("Gemma-3-4B-Instruct PREDICTIONS (prompt = GT labels for this image)")
        lines.append("-" * 40)
        for cat in gt_cats_for_prompt:
            display = _gt_display_name(cat)
            bbox = detected_items.get(display) or detected_items.get(cat)
            if bbox is not None:
                lines.append(f"  {display}: {_fmt_bbox(bbox)}")
            else:
                lines.append(f"  {display}: (not detected)")
        lines.append("")
        lines.append("IoU (Predicted vs Ground-Truth, by category)")
        lines.append("-" * 40)
        for cat in gt_cats_for_prompt:
            display = _gt_display_name(cat)
            v = iou_per_cat.get(display) or iou_per_cat.get(cat)
            s = f"{v:.3f}" if v is not None else "—"
            lines.append(f"  {display:30s}: {s}")
        if mean_iou is not None:
            lines.append(f"  {'mean':30s}: {mean_iou:.3f}")
        else:
            lines.append("  (no GT boxes for any category)")
        lines.append("")
        lines.append("=" * 60)

        report_text = "\n".join(lines)
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        def _draw_gt_bboxes(ax):
            """Draw GT bboxes on axis (all categories with 1 bbox)."""
            for cat_name, boxes in gt_single.items():
                display_name = _gt_display_name(cat_name)
                color = _color_for_cat(display_name)
                x1, y1, x2, y2 = boxes[0]
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor="none",
                ))
                ax.text(x1, y1 - 6, display_name, color="white", fontsize=9, weight="bold",
                        bbox=dict(facecolor=color, alpha=0.8))

        # --- bounding_box_comparison.png (GT + predictions) ---
        fig_bbox, axes_bbox = plt.subplots(3, 1, figsize=(10, 14))
        # Row 0: image + GT bboxes only
        axes_bbox[0].imshow(image)
        _draw_gt_bboxes(axes_bbox[0])
        axes_bbox[0].set_title("Ground truth bboxes", fontsize=10)
        axes_bbox[0].axis("off")
        # Row 1: image + predicted bboxes only
        axes_bbox[1].imshow(image)
        for item_name, (x1, y1, x2, y2) in detected_items.items():
            color = _color_for_cat(item_name)
            axes_bbox[1].add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            ))
            axes_bbox[1].text(x1, y1 - 6, item_name, color="white", fontsize=9, weight="bold",
                              bbox=dict(facecolor=color, alpha=0.8))
        axes_bbox[1].set_title("Predicted bboxes", fontsize=10)
        axes_bbox[1].axis("off")
        # Row 2: image + GT + predicted (both; GT and pred so we can compare)
        axes_bbox[2].imshow(image)
        for cat_name, boxes in gt_single.items():
            display_name = _gt_display_name(cat_name)
            x1, y1, x2, y2 = boxes[0]
            axes_bbox[2].add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none", linestyle="-",
            ))
            axes_bbox[2].text(x1, y1 - 6, display_name + " (GT)", color="white", fontsize=8, weight="bold",
                              bbox=dict(facecolor="green", alpha=0.8))
        for item_name, (x1, y1, x2, y2) in detected_items.items():
            color = _color_for_cat(item_name)
            axes_bbox[2].add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
            ))
            axes_bbox[2].text(x1, y1 - 6, item_name + " (pred)", color="white", fontsize=8, weight="bold",
                              bbox=dict(facecolor=color, alpha=0.8))
        axes_bbox[2].set_title("Ground truth (green solid) + Predicted (red dashed)", fontsize=10)
        axes_bbox[2].axis("off")
        iou_lines = ["IoU (Pred vs GT):"]
        for cat in gt_cats_for_prompt:
            display = _gt_display_name(cat)
            v = iou_per_cat.get(display) or iou_per_cat.get(cat)
            s = f"{v:.3f}" if v is not None else "—"
            iou_lines.append(f"  {display}={s}")
        if mean_iou is not None:
            iou_lines.append(f"  mean={mean_iou:.3f}")
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        fig_bbox.text(0.5, 0.02, "  ".join(iou_lines), ha="center", fontsize=9, family="monospace",
                      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        bbox_fig_path = os.path.join(output_dir, "bounding_box_comparison.png")
        plt.savefig(bbox_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig_bbox)

        tqdm.write(f"  [{_idx + 1}/{len(_image_paths)}] saved -> {output_dir}")
    except Exception as e:
        tqdm.write(f"  [{_idx + 1}/{len(_image_paths)}] FAILED {_img_name}: {e}")

# --- Single bbox detection (shoes) — first image only ---
if _image_paths:
    image_path = _image_paths[0]
    image = Image.open(image_path).convert("RGB")
    print(f"Single-shoe demo image size: {image.size}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Detect the shoes in this image and return the bounding box coordinates in the format [x1, y1, x2, y2]. Only return the coordinates, nothing else."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Model output: {output_text}")

    coords_match = re.findall(r'\[?\(?([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)?\]?', output_text)
    if coords_match:
        x1, y1, x2, y2 = map(float, coords_match[0])
        img_w, img_h = image.size[0], image.size[1]
        max_val = max(x1, y1, x2, y2)
        if max_val <= 1.5:
            x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        elif max_val <= 1000 and min(x1, y1, x2, y2) >= 0:
            x1, y1, x2, y2 = _bbox_1000_to_pixel(x1, y1, x2, y2, img_w, img_h, letterbox_1000=True)
        print(f"Extracted BBox (pixel): ({x1}, {y1}, {x2}, {y2})")
        fig, ax = plt.subplots(1, figsize=(10, 12))
        ax.imshow(image)
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none'))
        ax.text(x1, y1 - 5, 'shoe', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.7))
        ax.axis('off')
        ax.set_title("Detected shoe")
        plt.show()
    else:
        print("Could not parse coordinates from output")

print(f"Done. Processed {len(_image_paths)} images.")
print("Processed image names:")
for _p in _image_paths:
    print(f"  {os.path.basename(_p)}")
print(f"Outputs: {_output_base}")
if os.path.isdir(_output_base):
    _subs = [d for d in os.listdir(_output_base) if os.path.isdir(os.path.join(_output_base, d))]
    print(f"Subfolders: {len(_subs)} ({', '.join(sorted(_subs)[:5])}{'...' if len(_subs) > 5 else ''})")
