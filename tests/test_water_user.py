"""Test water detection v2 on the user-specified images."""
import os, sys, cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from features import extract_curvature_features, extract_all_geometry_features
from water_detection import detect_water

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
                coords[:, 0] *= w; coords[:, 1] *= h
                polygons.append(coords.astype(np.int32))
    return polygons

def run_on_image(img_path, label_path, depth_path, tag, idx):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERROR: Could not load {img_path}"); return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    depth_map = np.load(depth_path)
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    polygons = parse_yolo_label(label_path, (h, w))
    if not polygons:
        print(f"No labels in {label_path}"); return

    for pi, poly in enumerate(polygons):
        if poly.shape[0] < 3: continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)

        curvature_feats = extract_curvature_features(mask)
        geometry_feats = extract_all_geometry_features(mask, depth_map)
        water = detect_water(img_rgb, mask, depth_map=depth_map, curvature_features=curvature_feats)

        # ── 6-panel visualization ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.patch.set_facecolor('#0f172a')

        # Panel 1: Original + contour + bbox
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_c = img_rgb.copy()
        cv2.drawContours(img_c, contours, -1, (0, 255, 0), 2)
        x1, y1 = np.min(poly, axis=0); x2, y2 = np.max(poly, axis=0)
        cv2.rectangle(img_c, (x1, y1), (x2, y2), (255, 255, 0), 2)
        axes[0,0].imshow(img_c); axes[0,0].set_title("Original + Boundary", color='w', fontsize=12, fontweight='bold'); axes[0,0].axis('off')

        # Panel 2: Mask overlay
        ov = img_rgb.copy(); mc = np.zeros_like(img_rgb)
        mc[mask > 0] = [255, 100, 0] if not water['is_water'] else [0, 120, 255]
        ov = cv2.addWeighted(ov, 0.6, mc, 0.4, 0)
        axes[0,1].imshow(ov); axes[0,1].set_title("Mask Overlay", color='w', fontsize=12, fontweight='bold'); axes[0,1].axis('off')

        # Panel 3: Depth heatmap
        dn = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dc = cv2.cvtColor(cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        axes[0,2].imshow(dc); axes[0,2].contour(mask, levels=[0.5], colors='cyan', linewidths=1.5)
        axes[0,2].set_title("Depth Map + Contour", color='w', fontsize=12, fontweight='bold'); axes[0,2].axis('off')

        # Panel 4: Curvature profile
        if curvature_feats:
            keys = ['max_curvature','mean_curvature','p90_curvature','curvature_entropy','high_curvature_fraction','concave_fraction']
            vals = [curvature_feats.get(k, 0) for k in keys]
            labs = ['Max Curv','Mean Curv','P90 Curv','Entropy','High %','Concave %']
            cols = ['#f59e0b','#fb923c','#f97316','#06b6d4','#84cc16','#a78bfa']
            bars = axes[1,0].barh(labs, vals, color=cols, height=0.6)
            for b, v in zip(bars, vals):
                axes[1,0].text(b.get_width()+max(vals)*0.02, b.get_y()+b.get_height()/2, f'{v:.4f}', va='center', color='w', fontsize=8)
        axes[1,0].set_title("Curvature Profile", color='w', fontsize=12, fontweight='bold')
        axes[1,0].tick_params(colors='#94a3b8', labelsize=9); axes[1,0].set_facecolor('#1e293b')

        # Panel 5: Water cue scores (6 cues now)
        cue_names = ['Laplacian\nSmoothness', 'Texture\nVariance', 'Color\nDistrib.', 'Specular\nReflect.', 'Saturation\nUniform.', 'Depth\nInconsis.']
        cue_vals = [
            water['smoothness_score'], water['texture_score'], water['color_score'],
            water['specular_score'], water['saturation_score'],
            water['inconsistency_score'] if water['inconsistency_score'] is not None else 0.0,
        ]
        cue_cols = ['#22d3ee','#3b82f6','#8b5cf6','#ec4899','#a78bfa','#ef4444']
        bars2 = axes[1,1].bar(cue_names, cue_vals, color=cue_cols, width=0.55, edgecolor='white', linewidth=0.5)
        axes[1,1].set_ylim(0, 1.1)
        axes[1,1].axhline(y=0.35, color='#facc15', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (0.35)')
        axes[1,1].set_title("Water Detection Cues (v2)", color='w', fontsize=12, fontweight='bold')
        axes[1,1].tick_params(colors='#94a3b8', labelsize=7); axes[1,1].set_facecolor('#1e293b')
        axes[1,1].legend(fontsize=8, facecolor='#1e293b', edgecolor='#475569', labelcolor='white', loc='upper right')
        for b, v in zip(bars2, cue_vals):
            axes[1,1].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.3f}', ha='center', color='w', fontsize=8, fontweight='bold')

        # Panel 6: Verdict
        axes[1,2].set_facecolor('#1e293b'); axes[1,2].axis('off')
        prob = water['water_probability']; is_w = water['is_water']; conf = water['confidence_level']
        vc = '#3b82f6' if is_w else '#22c55e'
        vt = 'WATER DETECTED' if is_w else 'DRY POTHOLE'
        axes[1,2].text(0.5, 0.85, vt, transform=axes[1,2].transAxes, ha='center', fontsize=22, fontweight='bold', color=vc)
        axes[1,2].text(0.5, 0.70, f'Probability: {prob:.1%}', transform=axes[1,2].transAxes, ha='center', fontsize=16, color='white')
        cc = {'none':'#64748b','low':'#f59e0b','medium':'#fb923c','high':'#ef4444'}
        axes[1,2].text(0.5, 0.57, f'Confidence: {conf.upper()}', transform=axes[1,2].transAxes, ha='center', fontsize=13, color=cc.get(conf,'#94a3b8'), fontweight='bold')

        geo_lines = []
        if curvature_feats:
            geo_lines.append(f"Contour Length:   {curvature_feats.get('contour_length',0):.1f} px")
            geo_lines.append(f"Max Curvature:    {curvature_feats.get('max_curvature',0):.6f}")
        if geometry_feats:
            geo_lines.append(f"Bowl Depth:       {geometry_feats.get('mean_bowl_depth',0):.4f}")
            geo_lines.append(f"Normal Deviation: {geometry_feats.get('mean_normal_deviation',0):.2f} deg")
        geo_lines.append(f"Pothole Area:     {int(np.sum(mask))} px")
        for j, line in enumerate(geo_lines):
            axes[1,2].text(0.5, 0.42-j*0.08, line, transform=axes[1,2].transAxes, ha='center', fontsize=9, color='#cbd5e1', family='monospace')

        for ax in axes.flat:
            ax.set_facecolor('#1e293b')
            for s in ax.spines.values(): s.set_color('#334155')

        plt.suptitle(f'Water Detection v2 - {tag} (Pothole {pi+1}/{len(polygons)})', color='#f59e0b', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(OUT_DIR, f"v2_test_{idx+1}_pothole{pi+1}.png")
        plt.savefig(out_path, dpi=130, facecolor='#0f172a', bbox_inches='tight'); plt.close('all')

        print(f"\n{'='*60}")
        print(f"  {tag} - Pothole #{pi+1}/{len(polygons)}")
        print(f"{'='*60}")
        print(f"  VERDICT:           {'WATER' if is_w else 'DRY'}  (prob={prob:.1%}, conf={conf})")
        print(f"  Smoothness (Lap):  {water['smoothness_score']:.4f}")
        print(f"  Texture (rel):     {water['texture_score']:.4f}")
        print(f"  Color:             {water['color_score']:.4f}")
        print(f"  Specular:          {water['specular_score']:.4f}")
        print(f"  Saturation:        {water['saturation_score']:.4f}")
        print(f"  Inconsistency:     {water['inconsistency_score']}")
        print(f"  Saved: {out_path}")

images = [
    {
        "img": os.path.join(SCRIPT_DIR, "data1", "train", "images", "pic-1-_jpg.rf.49882cdb272111f43a6656b1494a4918.jpg"),
        "label": os.path.join(SCRIPT_DIR, "data1", "train", "labels", "pic-1-_jpg.rf.49882cdb272111f43a6656b1494a4918.txt"),
        "depth": os.path.join(SCRIPT_DIR, "depth_maps_1", "train", "pic-1-_jpg.rf.49882cdb272111f43a6656b1494a4918.npy"),
        "tag": "pic-1 (Rainy/Water)",
    },
    {
        "img": os.path.join(SCRIPT_DIR, "data1", "train", "images", "pic-9-_jpg.rf.10d9db0c8fac4eb5b01de9fc71dd19da.jpg"),
        "label": os.path.join(SCRIPT_DIR, "data1", "train", "labels", "pic-9-_jpg.rf.10d9db0c8fac4eb5b01de9fc71dd19da.txt"),
        "depth": os.path.join(SCRIPT_DIR, "depth_maps_1", "train", "pic-9-_jpg.rf.10d9db0c8fac4eb5b01de9fc71dd19da.npy"),
        "tag": "pic-9 (Standing Water)",
    },
]

print("Testing water detection v2 on user images...\n")
for idx, e in enumerate(images):
    run_on_image(e["img"], e["label"], e["depth"], e["tag"], idx)
print(f"\nDone! Results in: {OUT_DIR}")
