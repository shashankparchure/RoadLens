import os
import cv2
import glob
from segmentation import get_all_masks
from temporal_analysis import estimate_pothole_age, predict_severity_progression
from inference import get_depth_map
from features import extract_depth_features
from classifier import classify_severity

def test_batch3_on_images():
    print("="*60)
    print("Testing Batch 3: Temporal Analysis & Progression")
    print("="*60)
    
    # Grab a few test images from the dataset
    test_dir = r"data1\train\images"
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))[:3]
    
    if not image_paths:
        print("No images found in data1/train/images")
        return

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\n--- Processing: {img_name} ---")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print("Failed to load image.")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        masks = get_all_masks(img_path)
        if not masks:
            print("No potholes detected.")
            continue
            
        # Get depth map
        depth_map = get_depth_map(image)
        if masks[0].shape != depth_map.shape[:2]:
            depth_map = cv2.resize(depth_map, (masks[0].shape[1], masks[0].shape[0]), interpolation=cv2.INTER_LINEAR)
            
        for idx, mask in enumerate(masks):
            print(f"\n  Pothole #{idx+1}:")
            
            # Baseline severity (needed for progression prediction)
            depth_features = extract_depth_features(mask, depth_map)
            if depth_features is None:
                continue
            current_severity = classify_severity(depth_features)
            
            # 1. TEMPORAL ANALYSIS: Age Estimation
            age_estimate = estimate_pothole_age(mask, image_rgb)
            
            if age_estimate:
                print("    [Age Estimation]")
                print(f"      Category:      {age_estimate['age_category']} ({age_estimate['age_range']})")
                print(f"      Description:   {age_estimate['age_description']}")
                print(f"      Age Score:     {age_estimate['age_score']:.2f}")
                print(f"      - Edge Sharpness:     {age_estimate['edge_sharpness']:.2f} (0=smooth, 1=sharp)")
                print(f"      - Bndry Irregularity: {age_estimate['boundary_irregularity']:.2f} (0=smooth, 1=jagged)")
                print(f"      - Crack Texture:      {age_estimate['crack_texture_score']:.2f} (0=none, 1=heavy)")
                
                # 2. TEMPORAL ANALYSIS: Progression Prediction
                # We'll use a harsh weather context to see accelerated degradation
                progression = predict_severity_progression(
                    current_severity=current_severity,
                    age_estimate=age_estimate,
                    weather_context="freeze_thaw"
                )
                
                print("\n    [Severity Progression Prediction]")
                print(f"      Current State: {progression['current_severity']} (Score: {progression['current_score']:.2f})")
                print(f"      Weather:       {progression['weather_context']} (Multiplier: {progression['weather_factor']}x)")
                print(f"      Age Factor:    {progression['age_factor']:.2f}x speedup")
                print(f"      +30 Days:      {progression['prediction_30d']['severity']} (Score: {progression['prediction_30d']['score']:.2f})")
                print(f"      +60 Days:      {progression['prediction_60d']['severity']} (Score: {progression['prediction_60d']['score']:.2f})")
                print(f"      +90 Days:      {progression['prediction_90d']['severity']} (Score: {progression['prediction_90d']['score']:.2f})")
            else:
                print("    [Age Estimation] Failed to compute.")

if __name__ == "__main__":
    test_batch3_on_images()
