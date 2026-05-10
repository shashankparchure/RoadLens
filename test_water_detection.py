"""
Water Detection Test Script
Runs the full geometry + water detection pipeline on real dataset images
and generates a visual report.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from features import extract_curvature_features, extract_all_geometry_features
from water_detection import detect_water

# ── Paths ──
IMAGES_DIR = os.path.join(SCRIPT_DIR, "merged_dataset", "valid", "images")
LABELS_DIR = os.path.join(SCRIPT_DIR, "merged_dataset", "valid", "labels")
DEPTHS_DIR = os.path.join(SCRIPT_DIR, "depth_maps_global", "valid")
OUT_DIR = os.path.join(SCRIPT_DIR, "ml_results", "water_detection_test")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_yolo_label(label_path, img_shape):
    h, w = img_shape[:2]
    polygons = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                polygons.append(coords.astype(np.int32))
    return polygons


def run_test_on_image(img_name, idx):
    """Run full geometry + water detection on a single image."""
    basename = os.path.splitext(img_name)[0]
    img_path = os.path.join(IMAGES_DIR, img_name)
    label_path = os.path.join(LABELS_DIR, f"{basename}.txt")
    depth_path = os.path.join(DEPTHS_DIR, f"{basename}.npy")

    if not os.path.exists(label_path) or not os.path.exists(depth_path):
        return None

    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # Load depth map
    depth_map = np.load(depth_path)
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Parse YOLO labels → masks
    polygons = parse_yolo_label(label_path, (h, w))
    if not polygons:
        return None

    # Process first pothole in the image
    poly = polygons[0]
    if poly.shape[0] < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)

    # Extract geometry features
    curvature_feats = extract_curvature_features(mask)
    geometry_feats = extract_all_geometry_features(mask, depth_map)

    # Run water detection
    water_result = detect_water(
        img_rgb, mask,
        depth_map=depth_map,
        curvature_features=curvature_feats,
    )

    # ── Create visualization ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0f172a')

    # Panel 1: Original image with pothole outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img_rgb.copy()
    cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
    axes[0, 0].imshow(img_contour)
    axes[0, 0].set_title("Original + Pothole Boundary", color='white', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Mask overlay
    mask_overlay = img_rgb.copy()
    mask_colored = np.zeros_like(img_rgb)
    mask_colored[mask > 0] = [255, 100, 0]  # orange
    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
    axes[0, 1].imshow(mask_overlay)
    axes[0, 1].set_title("Segmentation Mask Overlay", color='white', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Panel 3: Depth heatmap (masked region highlighted)
    d_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    d_color_rgb = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
    axes[0, 2].imshow(d_color_rgb)
    axes[0, 2].contour(mask, levels=[0.5], colors='white', linewidths=1.5)
    axes[0, 2].set_title("Depth Map + Pothole Contour", color='white', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Panel 4: Curvature features bar chart
    if curvature_feats:
        display_keys = ['max_curvature', 'mean_curvature', 'p90_curvature',
                        'curvature_entropy', 'high_curvature_fraction', 'concave_fraction']
        display_vals = [curvature_feats.get(k, 0) for k in display_keys]
        short_labels = ['Max κ', 'Mean κ', 'P90 κ', 'Entropy', 'High κ%', 'Concave%']
        colors_bar = ['#f59e0b', '#fb923c', '#f97316', '#06b6d4', '#84cc16', '#a78bfa']
        bars = axes[1, 0].barh(short_labels, display_vals, color=colors_bar, height=0.6)
        axes[1, 0].set_title("Curvature Profile", color='white', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(colors='#94a3b8', labelsize=9)
        axes[1, 0].set_facecolor('#1e293b')
        for bar, val in zip(bars, display_vals):
            axes[1, 0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                          f'{val:.4f}', va='center', color='white', fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, 'Curvature N/A', transform=axes[1, 0].transAxes,
                        ha='center', va='center', color='#ef4444', fontsize=14)
        axes[1, 0].set_facecolor('#1e293b')

    # Panel 5: Water detection cue breakdown
    cue_names = ['Texture\nVariance', 'Color\nDistribution', 'Specular\nReflection', 'Depth-Appearance\nInconsistency']
    cue_values = [
        water_result['texture_score'],
        water_result['color_score'],
        water_result['specular_score'],
        water_result['inconsistency_score'] if water_result['inconsistency_score'] is not None else 0.0,
    ]
    cue_colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#ef4444']
    bars2 = axes[1, 1].bar(cue_names, cue_values, color=cue_colors, width=0.6, edgecolor='white', linewidth=0.5)
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].axhline(y=0.45, color='#facc15', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
    axes[1, 1].set_title("Water Detection Cue Scores", color='white', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(colors='#94a3b8', labelsize=8)
    axes[1, 1].set_facecolor('#1e293b')
    axes[1, 1].legend(fontsize=8, facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
    for bar, val in zip(bars2, cue_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{val:.3f}', ha='center', color='white', fontsize=9, fontweight='bold')

    # Panel 6: Final verdict card
    axes[1, 2].set_facecolor('#1e293b')
    axes[1, 2].axis('off')

    prob = water_result['water_probability']
    is_water = water_result['is_water']
    confidence = water_result['confidence_level']

    # Verdict background
    verdict_color = '#3b82f6' if is_water else '#22c55e'
    verdict_text = '💧 WATER DETECTED' if is_water else '🛣️ DRY POTHOLE'

    axes[1, 2].text(0.5, 0.82, verdict_text,
                    transform=axes[1, 2].transAxes, ha='center', va='center',
                    fontsize=20, fontweight='bold', color=verdict_color)

    axes[1, 2].text(0.5, 0.62, f'Probability: {prob:.1%}',
                    transform=axes[1, 2].transAxes, ha='center', va='center',
                    fontsize=16, color='white')

    conf_colors = {'low': '#94a3b8', 'medium': '#f59e0b', 'high': '#ef4444'}
    axes[1, 2].text(0.5, 0.47, f'Confidence: {confidence.upper()}',
                    transform=axes[1, 2].transAxes, ha='center', va='center',
                    fontsize=13, color=conf_colors.get(confidence, '#94a3b8'),
                    fontweight='bold')

    # Geometry summary
    if geometry_feats:
        geo_lines = [
            f"Contour Length: {geometry_feats.get('contour_length', 0):.1f} px",
            f"Mean Bowl Depth: {geometry_feats.get('mean_bowl_depth', 0):.4f}",
            f"Normal Deviation: {geometry_feats.get('mean_normal_deviation', 0):.1f}°",
        ]
        for j, line in enumerate(geo_lines):
            axes[1, 2].text(0.5, 0.30 - j * 0.10, line,
                          transform=axes[1, 2].transAxes, ha='center', va='center',
                          fontsize=10, color='#cbd5e1', family='monospace')

    axes[1, 2].text(0.5, 0.02, img_name,
                    transform=axes[1, 2].transAxes, ha='center', va='center',
                    fontsize=7, color='#475569')

    # Style all panels
    for ax in axes.flat:
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    plt.suptitle(f'Water Detection Analysis — Image #{idx+1}',
                 color='#f59e0b', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, f"water_test_{idx+1}_{basename[:40]}.png")
    plt.savefig(out_path, dpi=130, facecolor='#0f172a', bbox_inches='tight')
    plt.close('all')

    print(f"\n{'='*60}")
    print(f"Image: {img_name}")
    print(f"{'='*60}")
    print(f"  Water Detected:   {'YES' if is_water else 'NO'}")
    print(f"  Probability:      {prob:.4f} ({prob:.1%})")
    print(f"  Confidence:       {confidence}")
    print(f"  Texture Score:    {water_result['texture_score']:.4f}")
    print(f"  Color Score:      {water_result['color_score']:.4f}")
    print(f"  Specular Score:   {water_result['specular_score']:.4f}")
    print(f"  Inconsistency:    {water_result['inconsistency_score']}")
    if curvature_feats:
        print(f"  Max Curvature:    {curvature_feats['max_curvature']:.6f}")
        print(f"  Mean Curvature:   {curvature_feats['mean_curvature']:.6f}")
        print(f"  Contour Length:   {curvature_feats['contour_length']:.1f} px")
    if geometry_feats:
        print(f"  Bowl Depth:       {geometry_feats.get('mean_bowl_depth', 'N/A')}")
        print(f"  Normal Deviation: {geometry_feats.get('mean_normal_deviation', 'N/A')}°")
    print(f"  Saved: {out_path}")

    return water_result


def main():
    # Pick 3 diverse images from different sources
    test_images = []
    image_files = sorted(os.listdir(IMAGES_DIR))

    # Find one kaggle, one india, one pothole600
    kaggle = [f for f in image_files if f.startswith('kaggle')]
    india = [f for f in image_files if f.startswith('rdd_india')]
    japan = [f for f in image_files if f.startswith('rdd_japan')]

    if kaggle:
        test_images.append(kaggle[0])
    if india:
        test_images.append(india[0])
    if japan:
        test_images.append(japan[0])

    # Fill remaining from any source
    for f in image_files:
        if f not in test_images:
            test_images.append(f)
        if len(test_images) >= 3:
            break

    print(f"Testing water detection on {len(test_images)} images...")
    print(f"Output directory: {OUT_DIR}\n")

    results = []
    for idx, img_name in enumerate(test_images):
        result = run_test_on_image(img_name, idx)
        if result:
            results.append((img_name, result))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results)} images analyzed")
    print(f"{'='*60}")
    water_count = sum(1 for _, r in results if r['is_water'])
    print(f"  Water detected:  {water_count}/{len(results)}")
    print(f"  Avg probability: {np.mean([r['water_probability'] for _, r in results]):.4f}")
    print(f"  Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
