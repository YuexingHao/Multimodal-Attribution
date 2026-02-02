#!/usr/bin/env python3
"""
Summarize all report.txt files from each subfolder under output/.
Computes: (1) overall average mean IoU across images, (2) per-category average IoU.
Writes summary to output/summary_iou.txt and prints to stdout.
"""
import os
import re
from collections import defaultdict

# Script lives in output/; subfolders are siblings
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary_iou.txt")
REPORT_FILENAME = "report.txt"


def parse_report(path):
    """
    Parse report.txt; return (image_mean, category_iou_dict).
    category_iou_dict: category_name -> iou float (excludes 'mean' key).
    """
    with open(path) as f:
        text = f.read()
    # Find IoU section (from "IoU (Predicted..." until next "===" or end)
    iou_start = text.find("IoU (Predicted vs Ground-Truth, by category)")
    if iou_start == -1:
        return None, {}
    section = text[iou_start:]
    image_mean = None
    category_iou = {}
    for line in section.splitlines():
        # Lines like "  category    : 0.xxx" or "  mean        : 0.xxx"
        if ":" not in line:
            continue
        left, _, right = line.rstrip().rpartition(":")
        name = left.strip()
        val_str = right.strip()
        if not val_str.replace(".", "").isdigit():
            continue
        try:
            v = float(val_str)
        except ValueError:
            continue
        if name == "mean":
            image_mean = v
        else:
            category_iou[name] = v
    return image_mean, category_iou


def main():
    # Collect subdirs that contain report.txt
    subdirs = []
    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith(".") or name == os.path.basename(__file__):
            continue
        subpath = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(subpath):
            continue
        report_path = os.path.join(subpath, REPORT_FILENAME)
        if os.path.isfile(report_path):
            subdirs.append((name, report_path))

    if not subdirs:
        print("No report.txt found in any subfolder.")
        return

    # Aggregate
    image_means = []
    category_values = defaultdict(list)

    for subdir_name, report_path in subdirs:
        image_mean, category_iou = parse_report(report_path)
        if image_mean is not None:
            image_means.append(image_mean)
        for cat, iou in category_iou.items():
            category_values[cat].append(iou)

    # Build summary text
    lines = []
    lines.append("=" * 60)
    lines.append("FINDINGS SUMMARY (from report.txt in each output subfolder)")
    lines.append("=" * 60)
    lines.append("")

    overall_avg = None
    if image_means:
        overall_avg = sum(image_means) / len(image_means)
        min_mean, max_mean = min(image_means), max(image_means)
        lines.append("FINDINGS (averages)")
        lines.append("-" * 50)
        lines.append(f"  Total images with report: {len(image_means)}")
        lines.append(f"  Overall average mean IoU: {overall_avg:.4f}")
        lines.append(f"  Per-image mean IoU range: [{min_mean:.4f}, {max_mean:.4f}]")
        if category_values:
            cat_avgs = [(cat, sum(vals) / len(vals), len(vals)) for cat, vals in category_values.items()]
            cat_avgs.sort(key=lambda x: x[1], reverse=True)
            best_cat, best_avg, best_n = cat_avgs[0]
            worst_cat, worst_avg, worst_n = cat_avgs[-1]
            lines.append(f"  Best category (avg IoU):  {best_cat} = {best_avg:.4f}  (n={best_n})")
            lines.append(f"  Worst category (avg IoU): {worst_cat} = {worst_avg:.4f}  (n={worst_n})")
        lines.append("")
    else:
        lines.append("  No image means parsed.")
        lines.append("")

    lines.append("Overall average mean IoU (average of per-image means)")
    lines.append("-" * 50)
    if image_means:
        lines.append(f"  {overall_avg:.4f}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Per-category average IoU (across images that have this category)")
    lines.append("-" * 50)
    if category_values:
        for cat in sorted(category_values.keys()):
            vals = category_values[cat]
            avg = sum(vals) / len(vals)
            n = len(vals)
            lines.append(f"  {cat:40s}: {avg:.4f}  (n={n})")
    else:
        lines.append("  (no category IoU parsed)")
    lines.append("")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary_text)
    print(summary_text)
    print(f"Summary written to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
