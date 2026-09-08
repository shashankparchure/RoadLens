"""
Benchmark water detection (v3) on the Mendeley Water-Filled & Dry Potholes
dataset (Dib et al., 2023 — DOI: 10.17632/tp95cdvgm8.1).

The YOLO TXT annotations in this dataset use bounding boxes (class x_center
y_center width height) with a single class 0.  There is NO per-instance
water/dry label in the annotations, so we cannot compute ground-truth metrics
automatically.  Instead, this script:

  1. Runs `detect_water()` on every annotated pothole bbox across all 713
     images.
  2. Prints per-pothole verdicts in the terminal.
  3. Saves a 6-panel diagnostic image for each pothole (same layout as
     test_water_user.py).
  4. Produces summary charts: classification pie, confidence bar, probability
     histogram, per-cue box plot.

Usage:
    python tests/test_water_mendeley.py
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

# ── directories ─────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "water_detection_test_dataset", "water_test_dataset"
)
IMG_DIR = os.path.join(DATASET_ROOT, "IMG")
TXT_DIR = os.path.join(DATASET_ROOT, "TXT")

OUT_DIR = os.path.join(SCRIPT_DIR, "ml_results", "mendeley_dataset_tests")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  YOLO bbox parsing
# ═══════════════════════════════════════════════════════════════════════
def parse_yolo_bboxes(label_path, img_shape):
    """Parse YOLO bounding-box annotations.

    Format per line: <class_id> <x_center> <y_center> <width> <height>
    All values normalised to [0,1].

    Returns list of (class_id, x1, y1, x2, y2) in pixel coordinates.
    """
    h, w = img_shape[:2]
    bboxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bboxes.append((cls_id, x1, y1, x2, y2))
    return bboxes


# ═══════════════════════════════════════════════════════════════════════
#  6-panel visualisation (matches test_water_user.py style)
# ═══════════════════════════════════════════════════════════════════════
def save_panel(img_rgb, mask, water, tag, out_path):
    """Produce and save a 6-panel diagnostic image for one pothole."""
    h, w = img_rgb.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0f172a")

    # ── Panel 1: Original + contour + bbox ──
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_c = img_rgb.copy()
    cv2.drawContours(img_c, contours, -1, (0, 255, 0), 2)
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        cv2.rectangle(img_c, (xs.min(), ys.min()), (xs.max(), ys.max()), (255, 255, 0), 2)
    axes[0, 0].imshow(img_c)
    axes[0, 0].set_title("Original + Boundary", color="w", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # ── Panel 2: Mask overlay ──
    ov = img_rgb.copy()
    mc = np.zeros_like(img_rgb)
    mc[mask > 0] = [0, 120, 255] if water["is_water"] else [255, 100, 0]
    ov = cv2.addWeighted(ov, 0.6, mc, 0.4, 0)
    axes[0, 1].imshow(ov)
    axes[0, 1].set_title("Mask Overlay", color="w", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # ── Panel 3: Grayscale + contour (no depth map available) ──
    gray_rgb = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    axes[0, 2].imshow(gray_rgb)
    axes[0, 2].contour(mask, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[0, 2].set_title("Grayscale + Contour", color="w", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # ── Panel 4: Pixel statistics inside mask ──
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
                f"{v:.1f}",
                va="center", color="w", fontsize=8,
            )
    axes[1, 0].set_title("Pixel Stats (inside mask)", color="w", fontsize=12, fontweight="bold")
    axes[1, 0].tick_params(colors="#94a3b8", labelsize=9)
    axes[1, 0].set_facecolor("#1e293b")

    # ── Panel 5: Water cue scores ──
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

    # ── Panel 6: Verdict ──
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
        f"Pothole Area: {int(np.sum(mask))} px",
        f"Bbox Size:    {int(np.sum(mask > 0))} px",
    ]
    for j, line in enumerate(info):
        axes[1, 2].text(0.5, 0.40 - j * 0.08, line, transform=axes[1, 2].transAxes,
                         ha="center", fontsize=9, color="#cbd5e1", family="monospace")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        for s in ax.spines.values():
            s.set_color("#334155")

    plt.suptitle(f"Water Detection v3 — {tag}", color="#f59e0b",
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

    fig = plt.figure(figsize=(20, 14))
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
    ax4.set_title("Cue Score Distributions", color="w", fontsize=13, fontweight="bold")
    ax4.tick_params(colors="#94a3b8", labelsize=10)
    ax4.set_ylabel("Score", color="#94a3b8")
    ax4.axhline(y=0.5, color="#facc15", linestyle=":", linewidth=1, alpha=0.5)

    # ── 5. Water vs Dry cue comparison (mean ± std) ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1e293b")
    water_results = [r for r in results if r["is_water"]]
    dry_results = [r for r in results if not r["is_water"]]

    x_pos = np.arange(len(cue_keys))
    width = 0.35

    if water_results:
        water_means = [np.mean([r[k] for r in water_results]) for k in cue_keys]
        ax5.bar(x_pos - width / 2, water_means, width, color="#3b82f6",
                 label=f"Water (n={len(water_results)})", alpha=0.8, edgecolor="white", linewidth=0.5)

    if dry_results:
        dry_means = [np.mean([r[k] for r in dry_results]) for k in cue_keys]
        ax5.bar(x_pos + width / 2, dry_means, width, color="#22c55e",
                 label=f"Dry (n={len(dry_results)})", alpha=0.8, edgecolor="white", linewidth=0.5)

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
        f"Mendeley Pothole Water Detection Benchmark — {len(results)} potholes across 713 images",
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
    # Discover images
    img_files = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(img_files)} images in dataset.\n")

    all_results = []        # list of dicts per pothole
    total_potholes = 0
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

        bboxes = parse_yolo_bboxes(txt_path, (h, w))
        if not bboxes:
            continue

        for bi, (cls_id, x1, y1, x2, y2) in enumerate(bboxes):
            # Sanity: skip degenerate boxes
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            # Build rectangular mask from bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1

            water = detect_water(img_rgb, mask)

            tag = f"{stem}_p{bi + 1}"
            result = {
                "image": img_name,
                "pothole_idx": bi + 1,
                "tag": tag,
                **water,
            }
            all_results.append(result)
            total_potholes += 1

            # Terminal output
            verdict = "WATER" if water["is_water"] else "DRY"
            prob = water["water_probability"]
            conf = water["confidence_level"]
            print(
                f"  [{img_idx + 1:3d}/{len(img_files)}] {img_name:20s}  "
                f"pothole {bi + 1}/{len(bboxes)}  "
                f"-> {verdict:5s}  prob={prob:.3f}  conf={conf:6s}  "
                f"edge={water['edge_density_score']:.3f}  "
                f"grad={water['gradient_score']:.3f}  "
                f"color={water['color_score']:.3f}  "
                f"spec={water['specular_score']:.3f}  "
                f"sat={water['saturation_score']:.3f}"
            )

            # Save panel
            out_path = os.path.join(OUT_DIR, f"{tag}.png")
            save_panel(img_rgb, mask, water, tag, out_path)

    elapsed = time.time() - t_start

    # ── Terminal summary ────────────────────────────────────────────────
    water_count = sum(1 for r in all_results if r["is_water"])
    dry_count = total_potholes - water_count

    print("\n" + "=" * 72)
    print("  MENDELEY WATER DETECTION BENCHMARK — SUMMARY")
    print("=" * 72)
    print(f"  Images processed:    {len(img_files)}")
    print(f"  Potholes evaluated:  {total_potholes}")
    print(f"  Time elapsed:        {elapsed:.1f}s  ({elapsed / max(total_potholes, 1):.2f}s/pothole)")
    print(f"")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  WATER DETECTED:  {water_count:4d}  ({100 * water_count / max(total_potholes, 1):5.1f}%)  │")
    print(f"  │  DRY POTHOLE:     {dry_count:4d}  ({100 * dry_count / max(total_potholes, 1):5.1f}%)  │")
    print(f"  └─────────────────────────────────────┘")
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

    print(f"\n  Panel images saved to:  {OUT_DIR}")

    # ── Summary charts ──
    if all_results:
        save_summary_charts(all_results)

    print("\n  Done!\n")


if __name__ == "__main__":
    main()

