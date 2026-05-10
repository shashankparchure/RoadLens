"""
Depth Method Comparison: SfS vs Depth-Anything-V2
===================================================
Generates side-by-side visualizations comparing the Frankot-Chellappa
SfS reconstruction against the Depth-Anything-V2 monocular depth
estimation for pothole regions.

SfS captures fine-grained bowl curvature that monocular depth tends to
smooth over, while Depth-Anything provides better global relative depth.

Outputs:
    ml_results/sfs_vs_depth_anything/   (one PNG per image)
    ml_results/sfs_comparison_summary.csv

Usage:
    python scripts/compare_depth_methods.py [--max_images 20]
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shape_from_shading import reconstruct_depth_sfs, estimate_illumination

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(PROJECT_DIR, "ml_results", "sfs_vs_depth_anything")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset paths
VALID_IMG_DIR = os.path.join(PROJECT_DIR, "merged_dataset", "valid", "images")
VALID_LBL_DIR = os.path.join(PROJECT_DIR, "merged_dataset", "valid", "labels")
DEPTH_DIR = os.path.join(PROJECT_DIR, "depth_maps_merged", "valid")

if not os.path.isdir(VALID_IMG_DIR):
    VALID_IMG_DIR = os.path.join(PROJECT_DIR, "data1", "train", "images")
    VALID_LBL_DIR = os.path.join(PROJECT_DIR, "data1", "train", "labels")
    DEPTH_DIR = os.path.join(PROJECT_DIR, "depth_maps_1", "train")


def parse_first_mask(label_path, h, w):
    """Parse the first YOLO segmentation polygon into a binary mask."""
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
                return mask
    return None


def compute_metrics(depth_masked, sfs_masked):
    """Compute comparison metrics between DA-V2 depth and SfS height."""
    if len(depth_masked) == 0 or len(sfs_masked) == 0:
        return {'correlation': 0, 'mse': 0, 'structural_similarity': 0}

    # Normalize both to [0, 1]
    d = depth_masked.astype(np.float64)
    s = sfs_masked.astype(np.float64)

    d_min, d_max = d.min(), d.max()
    s_min, s_max = s.min(), s.max()

    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)

    if s_max > s_min:
        s = (s - s_min) / (s_max - s_min)
    else:
        s = np.zeros_like(s)

    # Pearson correlation
    if np.std(d) > 1e-8 and np.std(s) > 1e-8:
        correlation = float(np.corrcoef(d, s)[0, 1])
    else:
        correlation = 0.0

    # MSE
    mse = float(np.mean((d - s) ** 2))

    # Structural similarity (simplified — uses correlation + luminance + contrast)
    mu_d, mu_s = np.mean(d), np.mean(s)
    sig_d, sig_s = np.std(d), np.std(s)
    sig_ds = np.mean((d - mu_d) * (s - mu_s))

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_d * mu_s + C1) * (2 * sig_ds + C2)) / \
           ((mu_d ** 2 + mu_s ** 2 + C1) * (sig_d ** 2 + sig_s ** 2 + C2))

    return {
        'correlation': round(correlation, 4),
        'mse': round(mse, 6),
        'structural_similarity': round(float(ssim), 4),
    }


def generate_comparison_figure(img_rgb, depth_map, sfs_height, mask, metrics, tag, out_path):
    """Generate a 4-panel comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0f172a')

    h, w = mask.shape

    # ── Panel 1: Original + Mask ──
    overlay = img_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    axes[0].imshow(overlay)
    axes[0].set_title('Original + Mask', color='w', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # ── Panel 2: Depth-Anything (masked) ──
    depth_vis = np.zeros((h, w), dtype=np.float32)
    d_inside = depth_map[mask > 0]
    if len(d_inside) > 0:
        d_min, d_max = d_inside.min(), d_inside.max()
        if d_max > d_min:
            depth_vis[mask > 0] = (depth_map[mask > 0] - d_min) / (d_max - d_min)
    depth_colored = plt.cm.inferno(depth_vis)[:, :, :3]
    # Zero out outside mask
    depth_colored[mask == 0] = [0.07, 0.09, 0.16]  # dark bg
    axes[1].imshow(depth_colored)
    axes[1].set_title('Depth-Anything-V2 (masked)', color='w', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # ── Panel 3: SfS Height (masked) ──
    if sfs_height is not None:
        sfs_vis = np.zeros((h, w), dtype=np.float32)
        s_inside = sfs_height[mask > 0]
        if len(s_inside) > 0 and np.std(s_inside) > 1e-8:
            s_min, s_max = np.percentile(s_inside, 2), np.percentile(s_inside, 98)
            if s_max > s_min:
                sfs_vis[mask > 0] = np.clip(
                    (sfs_height[mask > 0] - s_min) / (s_max - s_min), 0, 1
                )
        sfs_colored = plt.cm.viridis(sfs_vis)[:, :, :3]
        sfs_colored[mask == 0] = [0.07, 0.09, 0.16]
        axes[2].imshow(sfs_colored)
        axes[2].set_title('SfS Reconstruction (masked)', color='w', fontsize=11, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'SfS Failed', transform=axes[2].transAxes,
                    ha='center', va='center', fontsize=16, color='#ef4444')
        axes[2].set_title('SfS Reconstruction', color='w', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # ── Panel 4: Difference Map + Metrics ──
    if sfs_height is not None:
        # Normalize both inside mask
        d_norm = np.zeros((h, w), dtype=np.float64)
        s_norm = np.zeros((h, w), dtype=np.float64)
        di = depth_map[mask > 0].astype(np.float64)
        si = sfs_height[mask > 0].astype(np.float64)

        if di.max() > di.min():
            d_norm[mask > 0] = (di - di.min()) / (di.max() - di.min())
        if si.max() > si.min():
            s_norm[mask > 0] = (si - si.min()) / (si.max() - si.min())

        diff = np.abs(d_norm - s_norm)
        diff[mask == 0] = 0
        diff_colored = plt.cm.RdYlGn_r(diff / max(diff.max(), 0.01))[:, :, :3]
        diff_colored[mask == 0] = [0.07, 0.09, 0.16]
        axes[3].imshow(diff_colored)

        # Overlay metrics text
        metrics_text = 'Corr: {}\nMSE: {}\nSSIM: {}'.format(
            metrics['correlation'], metrics['mse'], metrics['structural_similarity']
        )
        axes[3].text(0.05, 0.95, metrics_text, transform=axes[3].transAxes,
                    fontsize=9, color='white', va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.8))
    else:
        axes[3].text(0.5, 0.5, 'N/A', transform=axes[3].transAxes,
                    ha='center', va='center', fontsize=16, color='#64748b')

    axes[3].set_title('Difference Map', color='w', fontsize=11, fontweight='bold')
    axes[3].axis('off')

    for ax in axes:
        ax.set_facecolor('#1e293b')
        for s in ax.spines.values():
            s.set_color('#334155')

    plt.suptitle('SfS vs Depth-Anything-V2 — {}'.format(tag),
                color='#f59e0b', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


def run_comparison(max_images=20):
    """Run the full SfS vs Depth-Anything comparison."""
    print("=" * 60)
    print("  SfS vs Depth-Anything-V2 Comparison")
    print("=" * 60)

    img_files = sorted([
        f for f in os.listdir(VALID_IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:max_images]

    print("  Processing {} images...".format(len(img_files)))

    summary_rows = []
    success_count = 0

    for idx, img_file in enumerate(img_files):
        base = os.path.splitext(img_file)[0]
        img_path = os.path.join(VALID_IMG_DIR, img_file)
        lbl_path = os.path.join(VALID_LBL_DIR, base + '.txt')
        depth_path = os.path.join(DEPTH_DIR, base + '.npy')

        if not os.path.exists(lbl_path) or not os.path.exists(depth_path):
            continue

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_bgr.shape[:2]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            depth_map = np.load(depth_path)
            if depth_map.shape != (h, w):
                depth_map = cv2.resize(depth_map, (w, h))

            mask = parse_first_mask(lbl_path, h, w)
            if mask is None or np.sum(mask) < 100:
                continue

            # ── Run SfS reconstruction ──
            sfs_height = reconstruct_depth_sfs(gray, mask)

            # ── Compute metrics ──
            if sfs_height is not None:
                metrics = compute_metrics(
                    depth_map[mask > 0],
                    sfs_height[mask > 0]
                )
                success_count += 1
            else:
                metrics = {'correlation': 0, 'mse': 0, 'structural_similarity': 0}

            # ── Generate figure ──
            out_path = os.path.join(RESULTS_DIR, 'comparison_{}.png'.format(base[:50]))
            generate_comparison_figure(
                img_rgb, depth_map, sfs_height, mask, metrics, base[:40], out_path
            )

            summary_rows.append({
                'image': base,
                'sfs_success': sfs_height is not None,
                'pothole_area_px': int(np.sum(mask)),
                **metrics,
            })

            if (idx + 1) % 5 == 0:
                print("  Processed {}/{}...".format(idx + 1, len(img_files)))

        except Exception as e:
            print("  Error on {}: {}".format(img_file, e))

    # ── Save summary CSV ──
    if summary_rows:
        csv_path = os.path.join(PROJECT_DIR, "ml_results", "sfs_comparison_summary.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print("\n  Summary CSV: {}".format(csv_path))

    print("  SfS success rate: {}/{} ({:.0f}%)".format(
        success_count, len(summary_rows), 100 * success_count / max(len(summary_rows), 1)))
    print("  Figures saved to: {}".format(RESULTS_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=20)
    args = parser.parse_args()
    run_comparison(max_images=args.max_images)
