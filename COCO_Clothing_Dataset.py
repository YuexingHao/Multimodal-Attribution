#!/usr/bin/env python
# coding: utf-8
"""
Easily load and transform datasets for object detection.
From: https://github.com/blinjrm/detection-datasets?tab=readme-ov-file

- If detection_datasets is not installed, data is loaded from existing data/train.json
  and data/val.json (no extra pip install needed).
- Download from S3 is skipped when data/train.json and data/val.json already exist.

Run once: pip install tqdm (and optionally detection_datasets)
Optional: pip install numpy pandas numba shap datasets seaborn
"""


import json
import os
import urllib
import zipfile
import pandas as pd
from tqdm import tqdm

# Optional: use detection_datasets if installed; otherwise load from JSON
try:
    from detection_datasets import DetectionDataset
    HAS_DETECTION_DATASETS = True
except ModuleNotFoundError:
    HAS_DETECTION_DATASETS = False

# Download from S3 (skipped if data already present)
RAW_TRAIN_IMAGES = 'https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip'
RAW_VAL_IMAGES = 'https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip'
RAW_TRAIN_ANNOTATIONS = 'https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json'
RAW_VAL_ANNOTATIONS = 'https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json'

DATA_DIR = os.path.join(os.getcwd(), 'data')
TRAIN_ANNOTATIONS = 'train.json'
VAL_ANNOTATIONS = 'val.json'


def download(url, target):
    """Download image and annotations."""
    if url.split('.')[-1] == 'zip':
        path, _ = urllib.request.urlretrieve(url=url)
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(target)
        os.remove(path)
    else:
        urllib.request.urlretrieve(url=url, filename=target)


def load_data_from_coco_json(annotations_path, images_dir):
    """Build a data_by_image-style DataFrame from COCO/Fashionpedia JSON (no detection_datasets)."""
    with open(annotations_path) as f:
        coco = json.load(f)
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_cat = {c["id"]: c["name"] for c in coco["categories"]}
    # Group annotations by image_id
    from collections import defaultdict
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)
    rows = []
    for img_id, img in id_to_img.items():
        anns = ann_by_img.get(img_id, [])
        file_name = img["file_name"]
        # Image may live in a subdir (e.g. train/ or test/)
        image_path = os.path.join(images_dir, file_name)
        if not os.path.isfile(image_path):
            image_path = os.path.join(os.path.dirname(images_dir), os.path.basename(images_dir), file_name)
        bboxes = []
        categories = []
        for a in anns:
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            bboxes.append(f"Bbox id {a['id']} [{x1}, {y1}, {x2}, {y2}]")
            categories.append(id_to_cat.get(a["category_id"], str(a["category_id"])))
        rows.append({
            "image_id": img_id,
            "image": image_path,
            "width": img["width"],
            "height": img["height"],
            "image_path": image_path,
            "bbox": bboxes,
            "category": categories,
            "objects": bboxes,
        })
    return pd.DataFrame(rows)


os.makedirs(DATA_DIR, exist_ok=True)

# Skip download if annotations (and optionally images) already exist
train_ann = os.path.join(DATA_DIR, TRAIN_ANNOTATIONS)
val_ann = os.path.join(DATA_DIR, VAL_ANNOTATIONS)
if not os.path.isfile(train_ann) or not os.path.isfile(val_ann):
    download(url=RAW_TRAIN_ANNOTATIONS, target=train_ann)
    download(url=RAW_VAL_ANNOTATIONS, target=val_ann)
if not os.path.isdir(os.path.join(DATA_DIR, "train")) or not os.path.isdir(os.path.join(DATA_DIR, "test")):
    download(url=RAW_TRAIN_IMAGES, target=DATA_DIR)
    download(url=RAW_VAL_IMAGES, target=DATA_DIR)

if HAS_DETECTION_DATASETS:
    config = {
        "dataset_format": "coco",
        "path": DATA_DIR,
        "splits": {
            "train": (TRAIN_ANNOTATIONS, "train"),
            "val": (VAL_ANNOTATIONS, "test"),
        },
    }
    dd = DetectionDataset().from_disk(**config)
    data_by_image = dd.get_data(index="image")
else:
    # Load from existing JSON (no detection_datasets required)
    train_df = load_data_from_coco_json(train_ann, os.path.join(DATA_DIR, "train"))
    val_df = load_data_from_coco_json(val_ann, os.path.join(DATA_DIR, "test"))
    data_by_image = pd.concat([train_df, val_df], ignore_index=True)

# Image corresponding to notebook lines 4-8: data_by_image.iloc[103] -> fb0046f5396fc4731f4dda322433f18c.jpg
TARGET_IMAGE_FILENAME = "fb0046f5396fc4731f4dda322433f18c.jpg"


def get_item_for_image(data_by_image, filename):
    """Get the row for the given image filename (e.g. fb0046f5396fc4731f4dda322433f18c.jpg)."""
    mask = data_by_image["image_path"].astype(str).str.endswith(filename)
    if mask.any():
        return data_by_image.loc[mask].iloc[0]
    return data_by_image.iloc[min(103, len(data_by_image) - 1)]  # fallback

print(data_by_image.head())

# Get item for target image (fb0046f5396fc4731f4dda322433f18c.jpg, notebook iloc[103])
first_item = get_item_for_image(data_by_image, TARGET_IMAGE_FILENAME)
print(first_item)

print("Keys:", first_item.keys())
print("\nImage ID:", first_item.get('image_id'))
print("Image:", first_item.get('image'))
print("Width:", first_item.get('width'))
print("Height:", first_item.get('height'))
print("Objects:", first_item.get('objects'))


# Flatten all category lists and get unique categories
all_categories = []
for cats in data_by_image['category']:
    if isinstance(cats, list):
        all_categories.extend(cats)
    else:
        all_categories.append(cats)

unique_categories = set(all_categories)
print(f"Total number of unique categories: {len(unique_categories)}")
print(f"\nCategories: {sorted(unique_categories)}")

# Count occurrences of each category
from collections import Counter
category_counts = Counter(all_categories)
print(f"\nCategory distribution:")
for category, count in category_counts.most_common():
    print(f"  {category}: {count}")


from PIL import Image
import matplotlib.pyplot as plt

# Get item for target image (fb0046f5396fc4731f4dda322433f18c.jpg, notebook iloc[103])
first_item = get_item_for_image(data_by_image, TARGET_IMAGE_FILENAME)

# Load the image from the path
image_path = first_item['image_path']
print(f"Loading image from: {image_path}")

img = Image.open(image_path)

# Display the image
plt.figure(figsize=(10, 12))
plt.imshow(img)
plt.axis('off')
plt.title(f"Image ID: {first_item.name}, Size: {first_item['width']}x{first_item['height']}")
plt.show()

# Print details
print(f"\nImage ID: {first_item.name}")
print(f"Size: {first_item['width']}x{first_item['height']}")
print(f"Categories: {first_item['category']}")
print(f"Number of bboxes: {len(first_item['bbox'])}")


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import re

# Load image (target: fb0046f5396fc4731f4dda322433f18c.jpg)
first_item = get_item_for_image(data_by_image, TARGET_IMAGE_FILENAME)
img = Image.open(first_item['image_path'])

# Create figure
fig, ax = plt.subplots(1, figsize=(12, 14))
ax.imshow(img)

# Draw bounding boxes
for bbox, category in zip(first_item['bbox'], first_item['category']):
    # Parse the bbox string: "Bbox id XXXXX [x1, y1, x2, y2]"
    bbox_str = str(bbox)
    coords = re.findall(r'\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', bbox_str)[0]
    x1, y1, x2, y2 = map(float, coords)
    
    # Draw rectangle
    rect = patches.Rectangle(
        (x1, y1), 
        x2 - x1, 
        y2 - y1,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label
    ax.text(x1, y1 - 5, category, 
            color='white', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.7))

ax.axis('off')
plt.title(f"Image {first_item.name} - Fashion Items")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from PIL import Image
import re

# Load image (target: fb0046f5396fc4731f4dda322433f18c.jpg)
first_item = get_item_for_image(data_by_image, TARGET_IMAGE_FILENAME)
img = Image.open(first_item['image_path'])

# Extract and display each bounding box
cropped_images = []

for i, (bbox, category) in enumerate(zip(first_item['bbox'], first_item['category'])):
    # Parse the bbox string: "Bbox id XXXXX [x1, y1, x2, y2]"
    bbox_str = str(bbox)
    coords = re.findall(r'\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', bbox_str)[0]
    x1, y1, x2, y2 = map(float, coords)
    
    # Crop the image to this bounding box
    cropped = img.crop((x1, y1, x2, y2))
    
    # Store the cropped image with its label
    cropped_images.append({
        'image': cropped,
        'label': category,
        'bbox': (x1, y1, x2, y2),
        'index': i
    })

# Display all cropped images in a grid
n_boxes = len(cropped_images)
cols = 3
rows = (n_boxes + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
axes = axes.flatten() if n_boxes > 1 else [axes]

for idx, item in enumerate(cropped_images):
    axes[idx].imshow(item['image'])
    axes[idx].set_title(f"{item['index']}: {item['label']}", fontsize=12)
    axes[idx].axis('off')

# Hide empty subplots
for idx in range(n_boxes, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.suptitle(f"Image {first_item.name} - Individual Items", y=1.00, fontsize=14)
plt.show()

print(f"\nExtracted {len(cropped_images)} items:")
for item in cropped_images:
    print(f"  {item['index']}: {item['label']} - BBox: {item['bbox']}")




from PIL import Image
import torch
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ============ MODEL INITIALIZATION ============
# Load the model and processor (add this at the beginning)
model_name = "Qwen/Qwen2-VL-7B-Instruct"  # or your specific model path

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # or torch.float16
    device_map="auto"
)
print("Model loaded successfully!")


# ============ HELPER FUNCTIONS ============
def get_image_categories(data_by_image, image_path):
    """
    Get unique categories for a specific image from ground truth
    """
    # Filter data for this specific image
    image_data = data_by_image[data_by_image['image_path'] == image_path]
    
    # Collect all categories for this image
    all_cats = []
    for cats in image_data['category'].values:
        if isinstance(cats, list):
            all_cats.extend(cats)
        else:
            all_cats.append(cats)
    
    # Return unique categories
    return list(set(all_cats))

def create_detection_prompt(image, categories):
    """
    Create a detection prompt with ground truth categories
    """
    # Format categories as bullet points
    categories_text = "\n".join([f"- {cat}" for cat in sorted(categories)])
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": f"""Detect the following fashion items in this image and return their bounding box coordinates:
{categories_text}

For each item found, return in this format:
item_name: [x1, y1, x2, y2]

Only return the coordinates, nothing else."""
                },
            ],
        }
    ]
    return messages

def denormalize_bbox(bbox, img_width, img_height):
    """
    Convert normalized coordinates (0-1) to pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Check if coordinates are normalized (all values between 0 and 1)
    if all(0 <= coord <= 1 for coord in bbox):
        x1 = x1 * img_width
        y1 = y1 * img_height
        x2 = x2 * img_width
        y2 = y2 * img_height
    
    return (x1, y1, x2, y2)

def process_single_image(item, data_by_image, model, processor, visualize=True):
    """
    Process a single image and return predictions
    """
    # Load image
    image_path = item['image_path']
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Get ground truth categories for this image
    gt_categories = get_image_categories(data_by_image, image_path)
    
    print(f"\nProcessing: {image_path}")
    print(f"Image size: {img_width} x {img_height}")
    print(f"Ground truth categories: {gt_categories}")
    
    # Create the prompt with ground truth categories
    messages = create_detection_prompt(image, gt_categories)
    
    # Prepare for the model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"Model output: {output_text}")
    
    # Parse all detected items and coordinates
    detected_items = {}
    detected_items_normalized = {}
    
    # Create dynamic pattern based on ground truth categories
    category_pattern = '|'.join([re.escape(cat.lower()) for cat in gt_categories])
    pattern = rf'({category_pattern})[:\s]+\[?\(?(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)?\]?'
    matches = re.findall(pattern, output_text.lower())
    
    for match in matches:
        item_name = match[0]
        coords = tuple(map(float, match[1:5]))
        detected_items_normalized[item_name] = coords
        
        # Denormalize coordinates
        denorm_coords = denormalize_bbox(coords, img_width, img_height)
        detected_items[item_name] = denorm_coords
    
    print(f"Detected {len(detected_items)} items: {list(detected_items.keys())}")
    
    # Visualize if requested
    if visualize and detected_items:
        visualize_predictions(image, detected_items, gt_categories, image_path)
    
    return {
        'image_path': image_path,
        'gt_categories': gt_categories,
        'detected_items': detected_items,
        'detected_items_normalized': detected_items_normalized,
        'output_text': output_text
    }

def visualize_predictions(image, detected_items, gt_categories, image_path):
    """
    Visualize bounding box predictions
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define colors for different categories
    color_palette = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 
                     'pink', 'brown', 'lime', 'navy', 'teal', 'gold', 'coral']
    
    # Create color mapping for categories
    colors = {}
    for idx, cat in enumerate(sorted(gt_categories)):
        colors[cat.lower()] = color_palette[idx % len(color_palette)]
    
    # Display image
    ax.imshow(image)
    ax.set_title(f"Qwen2-VL Predictions ({len(detected_items)} items detected)\n{image_path}", 
                 fontsize=16, weight='bold', pad=20)
    
    # Draw bounding boxes
    for item_name, (x1, y1, x2, y2) in detected_items.items():
        color = colors.get(item_name, 'red')
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with better positioning
        label_y = y1 - 15 if y1 > 30 else y2 + 25
        
        ax.text(x1, label_y, item_name, 
                color='white', fontsize=13, weight='bold',
                bbox=dict(facecolor=color, alpha=0.9, edgecolor='white', linewidth=1.5, pad=3),
                verticalalignment='bottom' if y1 > 30 else 'top')
    
    # Remove axes
    ax.axis('off')
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=colors[cat.lower()], 
                                     edgecolor='white', 
                                     label=cat.capitalize())
                      for cat in sorted(gt_categories) if cat.lower() in detected_items]
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.show()


# ============ OPTION 1: PROCESS SINGLE IMAGE ============
# Load a specific image (target: fb0046f5396fc4731f4dda322433f18c.jpg)
print("="*60)
print("PROCESSING SINGLE IMAGE")
print("="*60)

first_item = get_item_for_image(data_by_image, TARGET_IMAGE_FILENAME)
result = process_single_image(first_item, data_by_image, model, processor, visualize=True)

# ============ OPTION 2: PROCESS ALL IMAGES ============
print("\n" + "="*60)
print("PROCESSING ALL IMAGES")
print("="*60)

# Get unique image paths
unique_image_paths = data_by_image['image_path'].unique()
print(f"Total unique images: {len(unique_image_paths)}")

# Store all results
all_results = []

# Process each unique image
for idx, image_path in enumerate(tqdm(unique_image_paths)):
    print(f"\n--- Processing image {idx+1}/{len(unique_image_paths)} ---")
    
    # Get the first row for this image (they all have the same image_path)
    item = data_by_image[data_by_image['image_path'] == image_path].iloc[0]
    
    try:
        result = process_single_image(
            item, 
            data_by_image, 
            model, 
            processor, 
            visualize=False  # Set to True if you want to see all visualizations
        )
        all_results.append(result)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        continue

# ============ SAVE RESULTS ============
# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
print(f"\nProcessed {len(results_df)} images successfully")

# Save results
results_df.to_pickle('detection_results.pkl')
print("Results saved to 'detection_results.pkl'")

# ============ ANALYZE RESULTS ============
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

# Count detection statistics
total_gt_items = sum(len(r['gt_categories']) for r in all_results)
total_detected = sum(len(r['detected_items']) for r in all_results)

print(f"Total ground truth items: {total_gt_items}")
print(f"Total detected items: {total_detected}")
print(f"Detection rate: {total_detected/total_gt_items*100:.2f}%")

# Category-wise statistics
category_stats = {}
for result in all_results:
    for cat in result['gt_categories']:
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'detected': 0}
        category_stats[cat]['total'] += 1
        if cat.lower() in result['detected_items']:
            category_stats[cat]['detected'] += 1

print("\nCategory-wise Detection Rates:")
for cat, stats in sorted(category_stats.items()):
    rate = stats['detected'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"  {cat}: {stats['detected']}/{stats['total']} ({rate:.2f}%)")





