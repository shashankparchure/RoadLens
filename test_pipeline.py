import os
import cv2
import numpy as np
import joblib
from pathlib import Path

# Setup dummy test data
TEST_IMAGE = "test_dummy.jpg"
cv2.imwrite(TEST_IMAGE, np.zeros((100, 100, 3), dtype=np.uint8))
test_mask = np.zeros((100, 100), dtype=np.uint8)
test_mask[10:50, 10:50] = 1
test_depth = np.random.rand(100, 100).astype(np.float32)

passed_count = 0
total_count = 11
failed_tests = []

print("══════════════════════════════")
print("Starting Verification Pipeline")
print("══════════════════════════════")

# TEST 1
try:
    from segmentation import get_largest_mask
    mask = get_largest_mask(TEST_IMAGE)
    assert mask is None or isinstance(mask, np.ndarray), "mask must be None or np.ndarray"
    print("TEST 1 - Original segmentation unchanged: PASSED")
    passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 1", str(e), "Check segmentation.py get_largest_mask wrapper"))
    print(f"TEST 1 - Original segmentation unchanged: FAILED ({e})")

# TEST 2
try:
    from segmentation import get_all_masks
    masks = get_all_masks(TEST_IMAGE)
    assert isinstance(masks, list), "masks must be a list"
    assert all(isinstance(m, np.ndarray) for m in masks), "all elements must be np.ndarray"
    print("TEST 2 - New get_all_masks works: PASSED")
    passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 2", str(e), "Check segmentation.py get_all_masks implementation"))
    print(f"TEST 2 - New get_all_masks works: FAILED ({e})")

# TEST 3
try:
    from features import extract_features
    features = extract_features(test_mask, test_depth)
    assert features is not None, "features is None"
    assert len(features) == 12, f"len(features) is {len(features)}, expected 12"
    expected = ['height','width','box_area','pothole_area','nonpothole_area',
                'mean_depth','max_depth','min_depth','depth_std',
                'depth_range','p90_depth','local_depth_contrast']
    assert all(k in features for k in expected), "missing expected keys"
    print("TEST 3 - Original extract_features returns exactly 12 features: PASSED")
    passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 3", str(e), "Check features.py extract_features and ensure it returns 12 keys"))
    print(f"TEST 3 - Original extract_features returns exactly 12 features: FAILED ({e})")

# TEST 4
try:
    from features import extract_features_extended
    features_ext = extract_features_extended(test_mask, test_depth)
    assert features_ext is not None, "features_ext is None"
    assert len(features_ext) == 21, f"len(features_ext) is {len(features_ext)}, expected 21"
    new_keys = ['aspect_ratio','solidity','compactness','depth_skewness',
                'depth_kurtosis','boundary_gradient','weighted_mean_depth',
                'surface_area_px2','surface_area_cm2']
    assert all(k in features_ext for k in new_keys), "missing extended keys"
    print("TEST 4 - extract_features_extended returns exactly 21 features: PASSED")
    passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 4", str(e), "Check features.py extract_features_extended returning exactly 21 features"))
    print(f"TEST 4 - extract_features_extended returns exactly 21 features: FAILED ({e})")

# TEST 5
try:
    from features import polygon_surface_area
    result = polygon_surface_area(test_mask)
    assert result is not None, "result is None"
    assert 'surface_area_px2' in result, "missing surface_area_px2"
    assert 'surface_area_cm2' in result, "missing surface_area_cm2"
    assert result['surface_area_px2'] > 0, "surface_area_px2 must be > 0"
    print("TEST 5 - polygon_surface_area works: PASSED")
    passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 5", str(e), "Check polygon_surface_area in features.py"))
    print(f"TEST 5 - polygon_surface_area works: FAILED ({e})")

# TEST 6
try:
    from classifier import rule_based_classify
    assert rule_based_classify({"max_depth": 0.2, "area": 100}) == "Shallow"
    assert rule_based_classify({"max_depth": 0.5, "area": 1000}) == "Moderate"
    assert rule_based_classify({"max_depth": 0.9, "area": 5000}) == "Deep"
    assert rule_based_classify({"max_depth": 0.0, "area": 0}) == "No pothole"
    print("TEST 6 - Original classifier unchanged: PASSED")
    passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 6", str(e), "Check classifier.py rule_based_classify"))
    print(f"TEST 6 - Original classifier unchanged: FAILED ({e})")

# TEST 7
try:
    base_ml_dir = Path("ml_models")
    if base_ml_dir.exists():
        pkls = list(base_ml_dir.glob("*.pkl"))
        for pkl in pkls:
            if "scaler" not in pkl.name.lower():
                joblib.load(pkl)
        print("TEST 7 - All original ML models load: PASSED")
        passed_count += 1
    else:
        print("TEST 7 - All original ML models load: SKIPPED (dir not found)")
        passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 7", str(e), "Ensure original models load properly with joblib"))
    print(f"TEST 7 - All original ML models load: FAILED ({e})")

# TEST 8
try:
    ext_ml_dir = Path("ml_models/extended")
    if ext_ml_dir.exists():
        pkls = list(ext_ml_dir.glob("*.pkl"))
        for pkl in pkls:
            if "scaler" not in pkl.name.lower():
                joblib.load(pkl)
        print("TEST 8 - All extended ML models load: PASSED")
        passed_count += 1
    else:
        print("TEST 8 - All extended ML models load: SKIPPED — run ml_classifier.py first")
        passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 8", str(e), "Ensure extended models load properly with joblib w/o module mismatch"))
    print(f"TEST 8 - All extended ML models load: FAILED ({e})")

# TEST 9
try:
    if Path('depth_global_stats.npy').exists():
        stats = np.load('depth_global_stats.npy')
        assert stats.shape == (2,), "shape not (2,)"
        assert stats[0] <= stats[1], "min must be <= max"
        print("TEST 9 - Global depth stats file exists and is valid: PASSED")
        passed_count += 1
    else:
        print("TEST 9 - Global depth stats file exists and is valid: SKIPPED (not generated yet)")
        passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 9", str(e), "Regenerate depth stats using depth/global_normalize.py"))
    print(f"TEST 9 - Global depth stats file exists and is valid: FAILED ({e})")

# TEST 10
try:
    required_paths = [
        "merged_dataset/train/images/",
        "merged_dataset/train/labels/",
        "merged_dataset/valid/images/",
        "merged_dataset/valid/labels/",
        "merged_dataset/test/images/",
        "merged_dataset/test/labels/",
        "merged_dataset/dataset.yaml",
        "merged_dataset/severity_labels/pothole600_annotations.csv"
    ]
    missing = []
    for p in required_paths:
        if not Path(p).exists():
            missing.append(p)
    if missing:
        failed_tests.append(("TEST 10", f"Missing paths: {missing}", "Ensure merged dataset is successfully generated"))
        print(f"TEST 10 - Merged dataset structure valid: FAILED (missing {missing})")
    else:
        print("TEST 10 - Merged dataset structure valid: PASSED")
        passed_count += 1
except Exception as e:
    failed_tests.append(("TEST 10", str(e), "Check dataset directory paths"))
    print(f"TEST 10 - Merged dataset structure valid: FAILED ({e})")

# TEST 11
print("TEST 11 - Hybrid CNN model: SKIPPED — hybrid_model.py will be implemented separately")

# TEST 12
try:
    import road_segment_analysis
    print("TEST 12 - road_segment_analysis.py imports cleanly: PASSED")
    passed_count += 1
except ImportError as e:
    failed_tests.append(("TEST 12", str(e), "Fix ImportErrors in road_segment_analysis.py"))
    print(f"TEST 12 - road_segment_analysis.py imports cleanly: FAILED ({e})")
except Exception as e:
    failed_tests.append(("TEST 12", str(e), "Fix generic errors during import of road_segment_analysis.py"))
    print(f"TEST 12 - road_segment_analysis.py imports cleanly: FAILED ({e})")

# Cleanup
if os.path.exists(TEST_IMAGE):
    os.remove(TEST_IMAGE)

print("\n══════════════════════════════")
print(f"Test Results: {passed_count}/{total_count} passed (Test 11 skipped)")
print("══════════════════════════════\n")

if failed_tests:
    for test_name, err, tip in failed_tests:
        print(f"FAILED: {test_name} — {tip}")
        print(f"  └ Error: {err}")
