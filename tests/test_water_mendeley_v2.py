"""
Benchmark water detection (v3) on the Mendeley Water-Filled & Dry Potholes
dataset — **v2: using YOLO segmentation masks instead of raw bounding boxes**.

v1 used the dataset's bounding-box annotations as rectangular masks, which
diluted water cues with surrounding dry road pixels and caused most potholes
to be classified as DRY (probability clustered around 25-30%).

v2 runs the project's YOLOv8 segmentation model on each image to obtain
tight pothole masks, then runs `detect_water()` on those masks.  If YOLO
detects no potholes in an image, we fall back to the dataset bbox.

Dataset: Dib et al., 2023 — DOI: 10.17632/tp95cdvgm8.1
  - 713 images, 1152 annotated potholes (bounding-box YOLO format)
  - Reported: ~594 water-filled, remainder dry

Usage:
    F:\\IITJ\\GitHub\\BTP\\btenv\\Scripts\\python.exe tests\\test_water_mendeley_v2.py
"""

import os
import sys
import time
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from water_detection import detect_water
from segmentation import MODEL, _extract_binary_masks

# ── directories ─────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "water_detection_test_dataset", "water_test_dataset"
)
IMG_DIR = os.path.join(DATASET_ROOT, "IMG")
TXT_DIR = os.path.join(DATASET_ROOT, "TXT")

OUT_DIR = os.path.join(SCRIPT_DIR, "ml_results", "mendeley_dataset_tests_v2")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  YOLO bbox parsing  (dataset ground-truth)
# ═══════════════════════════════════════════════════════════════════════
def parse_yolo_bboxes(label_path, img_shape):
    """Parse YOLO bounding-box annotations → list of (cls, x1, y1, x2, y2)."""
    h, w = img_shape[:2]
    bboxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = (float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4]))
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                bboxes.append((cls_id, x1, y1, x2, y2))
    return bboxes


# ═══════════════════════════════════════════════════════════════════════
#  Mask ↔ bbox matching  (IoU between YOLO-seg mask and GT bbox)
# ═══════════════════════════════════════════════════════════════════════
def mask_bbox_iou(mask, bbox):
    """Compute IoU between a binary mask and a (x1,y1,x2,y2) bounding box."""
    _, x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 1
    inter = np.sum(mask & box_mask)
    union = np.sum(mask | box_mask)
    return inter / max(union, 1)


def match_masks_to_bboxes(masks, bboxes, iou_threshold=0.15):
    """Match YOLO-seg masks to GT bounding boxes using IoU.

    Returns a list of (bbox, mask) tuples.  If a bbox has no matching
    segmentation mask above the threshold, a rectangular fallback mask
    is created from the bbox itself.
    """
    matched = []
    used_masks = set()

    for bbox in bboxes:
        best_iou = 0.0
        best_idx = -1
        for mi, m in enumerate(masks):
            if mi in used_masks:
                continue
            iou = mask_bbox_iou(m, bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = mi

        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.append((bbox, masks[best_idx], "yolo_seg", best_iou))
            used_masks.add(best_idx)
        else:
            # Fallback: create mask from bbox
            _, x1, y1, x2, y2 = bbox
            h, w = masks[0].shape[:2] if masks else (0, 0)
            if h == 0:
                continue
            fb_mask = np.zeros((h, w), dtype=np.uint8)
            fb_mask[y1:y2, x1:x2] = 1
            matched.append((bbox, fb_mask, "bbox_fallback", 0.0))

    return matched


# ═══════════════════════════════════════════════════════════════════════
#  6-panel visualisation
# ═══════════════════════════════════════════════════════════════════════
def save_panel(img_rgb, mask, water, tag, mask_source, iou, out_path):
    """Produce and save a 6-panel diagnostic image for one pothole."""
    h, w = img_rgb.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0f172a")

    # Panel 1: Original + contour + bbox
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_c = img_rgb.copy()
    cv2.drawContours(img_c, contours, -1, (0, 255, 0), 2)
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        cv2.rectangle(img_c, (xs.min(), ys.min()), (xs.max(), ys.max()), (255, 255, 0), 2)
    axes[0, 0].imshow(img_c)
    src_label = f"YOLO Seg (IoU={iou:.2f})" if mask_source == "yolo_seg" else "BBox Fallback"
    axes[0, 0].set_title(f"Original + Boundary [{src_label}]", color="w", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: Mask overlay
    ov = img_rgb.copy()
    mc = np.zeros_like(img_rgb)
    mc[mask > 0] = [0, 120, 255] if water["is_water"] else [255, 100, 0]
    ov = cv2.addWeighted(ov, 0.6, mc, 0.4, 0)
    axes[0, 1].imshow(ov)
    axes[0, 1].set_title("Mask Overlay", color="w", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: Grayscale + contour
    gray_rgb = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    axes[0, 2].imshow(gray_rgb)
    axes[0, 2].contour(mask, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[0, 2].set_title("Grayscale + Contour", color="w", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # Panel 4: Pixel statistics inside mask
    inside = img_rgb[mask > 0]
    if len(inside) > 0:
        labels = ["R mean", "G mean", "B mean", "R std", "G std", "B std"]
        vals = [
            inside[:, 0].mean(), inside[:, 1].mean(), inside[:, 2].mean(),
            inside[:, 0].std(),  inside[:, 1].std(),  inside[:, 2].std(),
        ]
        cols = ["#ef4444", "#22c55e", "#3b82f6", "#f87171", "#4ade80", "#60a5fa"]
        bars = axes[1, 0].barh(labels, vals, color=cols, height=0.6)
        for b, v in zip(bars, vals):
            axes[1, 0].text(
                b.get_width() + max(vals) * 0.02,
                b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", color="w", fontsize=8,
            )
    axes[1, 0].set_title("Pixel Stats (inside mask)", color="w", fontsize=12, fontweight="bold")
    axes[1, 0].tick_params(colors="#94a3b8", labelsize=9)
    axes[1, 0].set_facecolor("#1e293b")

    # Panel 5: Water cue scores
    cue_names = [
        "Edge\nDensity", "Gradient\nSmooth.", "Color\nDistrib.",
        "Specular\nReflect.", "Saturation\nUniform.", "Depth\nInconsis.",
    ]
    cue_vals = [
        water["edge_density_score"],
        water["gradient_score"],
        water["color_score"],
        water["specular_score"],
        water["saturation_score"],
        water["inconsistency_score"] if water["inconsistency_score"] is not None else 0.0,
    ]
    cue_cols = ["#22d3ee", "#3b82f6", "#8b5cf6", "#ec4899", "#a78bfa", "#ef4444"]
    bars2 = axes[1, 1].bar(cue_names, cue_vals, color=cue_cols, width=0.55,
                            edgecolor="white", linewidth=0.5)
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].axhline(y=0.35, color="#facc15", linestyle="--", linewidth=1.5,
                        alpha=0.7, label="Threshold (0.35)")
    axes[1, 1].set_title("Water Detection Cues (v3)", color="w", fontsize=12, fontweight="bold")
    axes[1, 1].tick_params(colors="#94a3b8", labelsize=7)
    axes[1, 1].set_facecolor("#1e293b")
    axes[1, 1].legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569",
                       labelcolor="white", loc="upper right")
    for b, v in zip(bars2, cue_vals):
        axes[1, 1].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f"{v:.3f}", ha="center", color="w", fontsize=8, fontweight="bold",
        )

    # Panel 6: Verdict
    axes[1, 2].set_facecolor("#1e293b")
    axes[1, 2].axis("off")
    prob = water["water_probability"]
    is_w = water["is_water"]
    conf = water["confidence_level"]

    vc = "#3b82f6" if is_w else "#22c55e"
    vt = "WATER DETECTED" if is_w else "DRY POTHOLE"
    axes[1, 2].text(0.5, 0.82, vt, transform=axes[1, 2].transAxes,
                     ha="center", fontsize=22, fontweight="bold", color=vc)
    axes[1, 2].text(0.5, 0.67, f"Probability: {prob:.1%}",
                     transform=axes[1, 2].transAxes, ha="center", fontsize=16, color="white")
    cc = {"none": "#64748b", "low": "#f59e0b", "medium": "#fb923c", "high": "#ef4444"}
    axes[1, 2].text(0.5, 0.54, f"Confidence: {conf.upper()}",
                     transform=axes[1, 2].transAxes, ha="center", fontsize=13,
                     color=cc.get(conf, "#94a3b8"), fontweight="bold")

    info = [
        f"Mask Source:   {mask_source}",
        f"Pothole Area:  {int(np.sum(mask))} px",
        f"Image Size:    {w}×{h}",
    ]
    for j, line in enumerate(info):
        axes[1, 2].text(0.5, 0.40 - j * 0.08, line, transform=axes[1, 2].transAxes,
                         ha="center", fontsize=9, color="#cbd5e1", family="monospace")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        for s in ax.spines.values():
            s.set_color("#334155")

    plt.suptitle(f"Water Detection v3 (Seg Mask) — {tag}", color="#f59e0b",
                  fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, facecolor="#0f172a", bbox_inches="tight")
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════
#  Summary charts
# ═══════════════════════════════════════════════════════════════════════
def save_summary_charts(results):
    """Generate aggregate summary visualisations."""

    water_count = sum(1 for r in results if r["is_water"])
    dry_count = len(results) - water_count
    probs = [r["water_probability"] for r in results]
    confs = [r["confidence_level"] for r in results]

    seg_count = sum(1 for r in results if r["mask_source"] == "yolo_seg")
    fb_count = len(results) - seg_count

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0f172a")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── 1. Pie chart: water vs dry ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1e293b")
    labels_pie = [f"Water ({water_count})", f"Dry ({dry_count})"]
    colours_pie = ["#3b82f6", "#22c55e"]
    wedges, texts, autotexts = ax1.pie(
        [water_count, dry_count], labels=labels_pie, colors=colours_pie,
        autopct="%1.1f%%", startangle=90, textprops={"color": "w", "fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax1.set_title("Classification Distribution", color="w", fontsize=13, fontweight="bold")

    # ── 2. Confidence level bar chart ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1e293b")
    conf_labels = ["none", "low", "medium", "high"]
    conf_counts = [confs.count(c) for c in conf_labels]
    conf_cols = ["#64748b", "#f59e0b", "#fb923c", "#ef4444"]
    bars = ax2.bar(conf_labels, conf_counts, color=conf_cols, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, conf_counts):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                  str(v), ha="center", color="w", fontsize=11, fontweight="bold")
    ax2.set_title("Confidence Level Distribution", color="w", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="#94a3b8", labelsize=10)
    ax2.set_ylabel("Count", color="#94a3b8")

    # ── 3. Probability histogram ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1e293b")
    ax3.hist(probs, bins=20, color="#8b5cf6", edgecolor="white", linewidth=0.5, alpha=0.9)
    ax3.axvline(x=0.35, color="#facc15", linestyle="--", linewidth=2, label="Threshold (0.35)")
    ax3.set_title("Water Probability Distribution", color="w", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Probability", color="#94a3b8")
    ax3.set_ylabel("Count", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8", labelsize=10)
    ax3.legend(fontsize=10, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    # ── 4. Per-cue box plots ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor("#1e293b")
    cue_keys = ["edge_density_score", "gradient_score", "color_score",
                 "specular_score", "saturation_score"]
    cue_labels = ["Edge Density", "Gradient", "Color", "Specular", "Saturation"]
    cue_data = [[r[k] for r in results] for k in cue_keys]
    cue_cols_box = ["#22d3ee", "#3b82f6", "#8b5cf6", "#ec4899", "#a78bfa"]

    bp = ax4.boxplot(cue_data, labels=cue_labels, patch_artist=True, widths=0.5,
                      medianprops=dict(color="white", linewidth=2),
                      whiskerprops=dict(color="#94a3b8"),
                      capprops=dict(color="#94a3b8"),
                      flierprops=dict(marker="o", markerfacecolor="#475569",
                                      markeredgecolor="#475569", markersize=3))
    for patch, col in zip(bp["boxes"], cue_cols_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax4.set_title("Cue Score Distributions (Seg Masks)", color="w", fontsize=13, fontweight="bold")
    ax4.tick_params(colors="#94a3b8", labelsize=10)
    ax4.set_ylabel("Score", color="#94a3b8")
    ax4.axhline(y=0.5, color="#facc15", linestyle=":", linewidth=1, alpha=0.5)

    # ── 5. Water vs Dry mean cue comparison ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1e293b")
    water_results = [r for r in results if r["is_water"]]
    dry_results = [r for r in results if not r["is_water"]]

    x_pos = np.arange(len(cue_keys))
    width = 0.35

    if water_results:
        water_means = [np.mean([r[k] for r in water_results]) for k in cue_keys]
        ax5.bar(x_pos - width / 2, water_means, width, color="#3b82f6",
                 label=f"Water (n={len(water_results)})", alpha=0.8,
                 edgecolor="white", linewidth=0.5)

    if dry_results:
        dry_means = [np.mean([r[k] for r in dry_results]) for k in cue_keys]
        ax5.bar(x_pos + width / 2, dry_means, width, color="#22c55e",
                 label=f"Dry (n={len(dry_results)})", alpha=0.8,
                 edgecolor="white", linewidth=0.5)

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(["Edge", "Grad", "Color", "Spec", "Satur"], fontsize=9)
    ax5.set_title("Mean Cues: Water vs Dry", color="w", fontsize=13, fontweight="bold")
    ax5.tick_params(colors="#94a3b8", labelsize=10)
    ax5.legend(fontsize=9, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")
    ax5.set_ylabel("Mean Score", color="#94a3b8")

    for ax in fig.axes:
        for s in ax.spines.values():
            s.set_color("#334155")

    plt.suptitle(
        f"Mendeley Water Detection v2 (YOLO Seg) — {len(results)} potholes "
        f"({seg_count} seg + {fb_count} bbox fallback)",
        color="#f59e0b", fontsize=15, fontweight="bold", y=0.99,
    )
    summary_path = os.path.join(OUT_DIR, "_summary_charts.png")
    plt.savefig(summary_path, dpi=150, facecolor="#0f172a", bbox_inches="tight")
    plt.close("all")
    print(f"\n  Summary charts saved: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    img_files = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(img_files)} images in dataset.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Using YOLO segmentation model for mask extraction.\n")

    all_results = []
    total_potholes = 0
    seg_used = 0
    bbox_fallback_used = 0
    yolo_no_detect = 0
    t_start = time.time()

    for img_idx, img_name in enumerate(img_files):
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)
        txt_path = os.path.join(TXT_DIR, stem + ".txt")

        if not os.path.exists(txt_path):
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  SKIP: cannot load {img_name}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # Parse GT bboxes
        bboxes = parse_yolo_bboxes(txt_path, (h, w))
        if not bboxes:
            continue

        # Run YOLO segmentation
        yolo_masks = _extract_binary_masks(
            image=img_bgr, model=MODEL, conf_threshold=0.25, min_area=100
        )

        if not yolo_masks:
            yolo_no_detect += 1

        # Match masks to GT bboxes
        matched = match_masks_to_bboxes(yolo_masks, bboxes, iou_threshold=0.15)

        for bbox, mask, mask_source, iou in matched:
            water = detect_water(img_rgb, mask)

            tag = f"{stem}_p{total_potholes + 1}"
            result = {
                "image": img_name,
                "tag": tag,
                "mask_source": mask_source,
                "iou": iou,
                **water,
            }
            all_results.append(result)
            total_potholes += 1

            if mask_source == "yolo_seg":
                seg_used += 1
            else:
                bbox_fallback_used += 1

            verdict = "WATER" if water["is_water"] else "DRY"
            prob = water["water_probability"]
            conf = water["confidence_level"]
            src = "SEG" if mask_source == "yolo_seg" else "BOX"
            print(
                f"  [{img_idx + 1:3d}/{len(img_files)}] {img_name:20s}  "
                f"[{src}]  "
                f"-> {verdict:5s}  prob={prob:.3f}  conf={conf:6s}  "
                f"edge={water['edge_density_score']:.3f}  "
                f"grad={water['gradient_score']:.3f}  "
                f"color={water['color_score']:.3f}  "
                f"spec={water['specular_score']:.3f}  "
                f"sat={water['saturation_score']:.3f}"
            )

            # Save panel
            out_path = os.path.join(OUT_DIR, f"{tag}.png")
            save_panel(img_rgb, mask, water, tag, mask_source, iou, out_path)

    elapsed = time.time() - t_start

    # ── Terminal summary ────────────────────────────────────────────────
    water_count = sum(1 for r in all_results if r["is_water"])
    dry_count = total_potholes - water_count

    print("\n" + "=" * 76)
    print("  MENDELEY WATER DETECTION v2 (YOLO SEG MASKS) — SUMMARY")
    print("=" * 76)
    print(f"  Images processed:      {len(img_files)}")
    print(f"  Potholes evaluated:    {total_potholes}")
    print(f"  Time elapsed:          {elapsed:.1f}s  ({elapsed / max(total_potholes, 1):.2f}s/pothole)")
    print()
    print(f"  ┌──── Mask Source ────────────────────────┐")
    print(f"  │  YOLO Seg masks:   {seg_used:4d}  ({100 * seg_used / max(total_potholes, 1):5.1f}%)  │")
    print(f"  │  BBox fallback:    {bbox_fallback_used:4d}  ({100 * bbox_fallback_used / max(total_potholes, 1):5.1f}%)  │")
    print(f"  │  YOLO no-detect:   {yolo_no_detect:4d}  images                │")
    print(f"  └────────────────────────────────────────┘")
    print()
    print(f"  ┌──── Classification ─────────────────────┐")
    print(f"  │  WATER DETECTED:   {water_count:4d}  ({100 * water_count / max(total_potholes, 1):5.1f}%)  │")
    print(f"  │  DRY POTHOLE:      {dry_count:4d}  ({100 * dry_count / max(total_potholes, 1):5.1f}%)  │")
    print(f"  └────────────────────────────────────────┘")
    print()

    # Confidence breakdown
    confs = [r["confidence_level"] for r in all_results]
    for c in ["high", "medium", "low", "none"]:
        n = confs.count(c)
        print(f"  Confidence {c:6s}: {n:4d}  ({100 * n / max(total_potholes, 1):5.1f}%)")

    # Mean cue scores
    print()
    cue_keys = ["edge_density_score", "gradient_score", "color_score",
                 "specular_score", "saturation_score"]
    cue_labels = ["Edge Density", "Gradient    ", "Color       ",
                   "Specular    ", "Saturation  "]
    for key, label in zip(cue_keys, cue_labels):
        vals = [r[key] for r in all_results]
        print(f"  {label}:  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # Probability stats
    probs = [r["water_probability"] for r in all_results]
    print()
    print(f"  Probability stats:  mean={np.mean(probs):.4f}  median={np.median(probs):.4f}  "
          f"std={np.std(probs):.4f}")

    # Compare seg vs bbox fallback performance
    seg_results = [r for r in all_results if r["mask_source"] == "yolo_seg"]
    fb_results = [r for r in all_results if r["mask_source"] == "bbox_fallback"]
    if seg_results:
        seg_water = sum(1 for r in seg_results if r["is_water"])
        seg_prob_mean = np.mean([r["water_probability"] for r in seg_results])
        print(f"\n  YOLO Seg masks:   {seg_water}/{len(seg_results)} water  "
              f"({100 * seg_water / len(seg_results):.1f}%)  mean_prob={seg_prob_mean:.3f}")
    if fb_results:
        fb_water = sum(1 for r in fb_results if r["is_water"])
        fb_prob_mean = np.mean([r["water_probability"] for r in fb_results])
        print(f"  BBox fallback:    {fb_water}/{len(fb_results)} water  "
              f"({100 * fb_water / len(fb_results):.1f}%)  mean_prob={fb_prob_mean:.3f}")

    print(f"\n  Panel images saved to:  {OUT_DIR}")

    # Summary charts
    if all_results:
        save_summary_charts(all_results)

    print("\n  Done!\n")


if __name__ == "__main__":
    main()

