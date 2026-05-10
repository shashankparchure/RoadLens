"""Test water detection v3 on all 4 user-tested images."""
import os, sys, cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_curvature_features, extract_all_geometry_features
from water_detection import detect_water

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'ml_results', 'water_detection_test')
os.makedirs(OUT, exist_ok=True)

def parse_label(p, shape):
    h, w = shape[:2]; polys = []
    with open(p) as f:
        for line in f:
            pts = line.strip().split()
            if len(pts) >= 3:
                c = np.array([float(x) for x in pts[1:]]).reshape(-1, 2)
                c[:, 0] *= w; c[:, 1] *= h
                polys.append(c.astype(np.int32))
    return polys

def run(tag, base, expected, idx):
    img = cv2.imread(os.path.join(ROOT, 'data1', 'train', 'images', base + '.jpg'))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    dm = np.load(os.path.join(ROOT, 'depth_maps_1', 'train', base + '.npy'))
    if dm.shape != (h, w): dm = cv2.resize(dm, (w, h))
    polys = parse_label(os.path.join(ROOT, 'data1', 'train', 'labels', base + '.txt'), (h, w))

    for pi, poly in enumerate(polys[:1]):  # First pothole only for quick test
        if poly.shape[0] < 3: continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        cf = extract_curvature_features(mask)
        gf = extract_all_geometry_features(mask, dm)
        wr = detect_water(rgb, mask, depth_map=dm, curvature_features=cf)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.patch.set_facecolor('#0f172a')

        # P1: Original
        ic = rgb.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ic, cnts, -1, (0, 255, 0), 2)
        x1, y1 = np.min(poly, axis=0); x2, y2 = np.max(poly, axis=0)
        cv2.rectangle(ic, (x1, y1), (x2, y2), (255, 255, 0), 2)
        axes[0, 0].imshow(ic)
        axes[0, 0].set_title('Original + Boundary', color='w', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # P2: Mask overlay
        ov = rgb.copy(); mc = np.zeros_like(rgb)
        mc[mask > 0] = [0, 120, 255] if wr['is_water'] else [255, 100, 0]
        ov = cv2.addWeighted(ov, 0.6, mc, 0.4, 0)
        axes[0, 1].imshow(ov)
        axes[0, 1].set_title('Mask Overlay', color='w', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # P3: Depth
        dn = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dc = cv2.cvtColor(cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        axes[0, 2].imshow(dc)
        axes[0, 2].contour(mask, levels=[0.5], colors='cyan', linewidths=1.5)
        axes[0, 2].set_title('Depth Map + Contour', color='w', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # P4: Curvature
        if cf:
            ks = ['max_curvature', 'mean_curvature', 'p90_curvature',
                  'curvature_entropy', 'high_curvature_fraction', 'concave_fraction']
            vs = [cf.get(k, 0) for k in ks]
            ls = ['Max', 'Mean', 'P90', 'Entropy', 'High%', 'Concave%']
            cs = ['#f59e0b', '#fb923c', '#f97316', '#06b6d4', '#84cc16', '#a78bfa']
            bs = axes[1, 0].barh(ls, vs, color=cs, height=0.6)
            for b, v in zip(bs, vs):
                axes[1, 0].text(b.get_width() + max(vs) * 0.02, b.get_y() + b.get_height() / 2,
                                '{:.4f}'.format(v), va='center', color='w', fontsize=8)
        axes[1, 0].set_title('Curvature Profile', color='w', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(colors='#94a3b8', labelsize=9)
        axes[1, 0].set_facecolor('#1e293b')

        # P5: Water cues (v3)
        cn = ['Edge\nDensity', 'Gradient\nSmooth', 'Specular\nRefl',
              'Color\nDist', 'Saturation\nUnif', 'Depth\nIncon']
        cvals = [
            wr['edge_density_score'], wr['gradient_score'], wr['specular_score'],
            wr['color_score'], wr['saturation_score'],
            wr['inconsistency_score'] if wr['inconsistency_score'] is not None else 0
        ]
        cc2 = ['#22d3ee', '#10b981', '#ec4899', '#8b5cf6', '#a78bfa', '#ef4444']
        bs2 = axes[1, 1].bar(cn, cvals, color=cc2, width=0.55, edgecolor='white', linewidth=0.5)
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].axhline(y=0.35, color='#facc15', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
        axes[1, 1].set_title('Water Cues (v3 Edge-Aware)', color='w', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(colors='#94a3b8', labelsize=7)
        axes[1, 1].set_facecolor('#1e293b')
        axes[1, 1].legend(fontsize=8, facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
        for b, v in zip(bs2, cvals):
            axes[1, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                            '{:.3f}'.format(v), ha='center', color='w', fontsize=8, fontweight='bold')

        # P6: Verdict
        axes[1, 2].set_facecolor('#1e293b')
        axes[1, 2].axis('off')
        p = wr['water_probability']
        iw = wr['is_water']
        co = wr['confidence_level']
        vcolor = '#3b82f6' if iw else '#22c55e'
        vtext = 'WATER DETECTED' if iw else 'DRY POTHOLE'
        axes[1, 2].text(0.5, 0.88, vtext, transform=axes[1, 2].transAxes,
                        ha='center', fontsize=22, fontweight='bold', color=vcolor)
        axes[1, 2].text(0.5, 0.74, 'Probability: {:.1%}'.format(p),
                        transform=axes[1, 2].transAxes, ha='center', fontsize=16, color='white')
        conf_colors = {'none': '#64748b', 'low': '#f59e0b', 'medium': '#fb923c', 'high': '#ef4444'}
        axes[1, 2].text(0.5, 0.62, 'Confidence: ' + co.upper(),
                        transform=axes[1, 2].transAxes, ha='center', fontsize=13,
                        color=conf_colors.get(co, '#94a3b8'), fontweight='bold')

        # Expected vs actual
        actual = 'WATER' if iw else 'DRY'
        correct = (actual == expected)
        check = 'CORRECT' if correct else 'WRONG'
        check_color = '#22c55e' if correct else '#ef4444'
        axes[1, 2].text(0.5, 0.50, 'Expected: {} | {}'.format(expected, check),
                        transform=axes[1, 2].transAxes, ha='center', fontsize=11,
                        color=check_color, fontweight='bold')

        info = []
        if cf:
            info.append('Contour: {:.1f} px'.format(cf.get('contour_length', 0)))
        if gf:
            info.append('Bowl Depth: {:.4f}'.format(gf.get('mean_bowl_depth', 0)))
        info.append('Area: {} px'.format(int(np.sum(mask))))
        for j, line in enumerate(info):
            axes[1, 2].text(0.5, 0.38 - j * 0.08, line, transform=axes[1, 2].transAxes,
                            ha='center', fontsize=9, color='#cbd5e1', family='monospace')

        for ax in axes.flat:
            ax.set_facecolor('#1e293b')
            for s in ax.spines.values(): s.set_color('#334155')

        plt.suptitle('Water Detection v3 - {} (Expected: {})'.format(tag, expected),
                     color='#f59e0b', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        op = os.path.join(OUT, 'v3_{}.png'.format(tag.replace(' ', '_').replace('/', '_')))
        plt.savefig(op, dpi=130, facecolor='#0f172a', bbox_inches='tight')
        plt.close('all')

        verdict = 'WATER' if iw else 'DRY'
        match = 'OK' if verdict == expected else 'MISMATCH'
        print('{} P#1: {} prob={:.1%} [expected={}] {} | edge={:.3f} grad={:.3f} spec={:.3f} col={:.3f} sat={:.3f} incon={}'.format(
            tag, verdict, p, expected, match,
            wr['edge_density_score'], wr['gradient_score'], wr['specular_score'],
            wr['color_score'], wr['saturation_score'], wr['inconsistency_score']))
        print('  -> ' + op)

# Test all 4 images with ground truth
tests = [
    ('pic-1 water',  'pic-1-_jpg.rf.49882cdb272111f43a6656b1494a4918',  'WATER'),
    ('pic-9 water',  'pic-9-_jpg.rf.10d9db0c8fac4eb5b01de9fc71dd19da',  'WATER'),
    ('pic-52 wet',   'pic-52-_jpg.rf.d8822e3b6a7c8fe4c73543cd7d7ae9cd', 'WATER'),
    ('pic-65 dry',   'pic-65-_jpg.rf.e602ed35d690902722b26561dd3f9684', 'DRY'),
]

print('Water Detection v3 — Testing 4 images with ground truth\n')
for i, (tag, base, expected) in enumerate(tests):
    run(tag, base, expected, i)
print('\nDone!')
