"""Quick test to reproduce and debug the /analyze 500 error."""
import cv2
import numpy as np
import json
import glob
import traceback

from api import compute_depth_slice, build_depth_annotated_image
from segmentation import get_all_masks
from inference import get_depth_map

# Find a test image
test_imgs = glob.glob("data1/*.jpg") + glob.glob("data1/*.png") + glob.glob("output/*.jpg")
if not test_imgs:
    test_imgs = glob.glob("*.jpg") + glob.glob("*.png")
print(f"Found {len(test_imgs)} test images")
if not test_imgs:
    print("No test images found")
    exit()

img_path = test_imgs[0]
print(f"Using: {img_path}")

img = cv2.imread(img_path)
if img is None:
    print("Cannot read image")
    exit()

masks = get_all_masks(img_path)
print(f"Got {len(masks)} masks")

depth = get_depth_map(img)
if depth.shape != img.shape[:2]:
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
print(f"Depth shape: {depth.shape}")

for i, m in enumerate(masks):
    try:
        profile = compute_depth_slice(m, depth)
        if profile:
            pts = profile["slicePoints"]
            json.dumps(profile)
            print(f"Mask {i}: OK, {len(pts)} points, bowl={profile['bowlDepth']}")
        else:
            print(f"Mask {i}: No profile (returned None)")
    except Exception as e:
        print(f"Mask {i}: ERROR: {e}")
        traceback.print_exc()

# Test annotated image
try:
    mask_records = []
    for i, m in enumerate(masks):
        dp = compute_depth_slice(m, depth)
        rec = {
            "consensusSeverity": "Deep",
            "id": i + 1,
            "bbox": None,
            "depthProfile": dp,
        }
        mask_records.append((m, rec))
    annotated = build_depth_annotated_image(img, depth, mask_records)
    print(f"Annotated image shape: {annotated.shape}")
except Exception as e:
    print(f"Annotated image ERROR: {e}")
    traceback.print_exc()

print("DONE")
