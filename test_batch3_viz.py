import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from segmentation import get_all_masks
from temporal_analysis import estimate_pothole_age, predict_severity_progression
from inference import get_depth_map
from features import extract_depth_features
from classifier import classify_severity

def test_batch3_with_viz():
    print("="*60)
    print("Testing Batch 3: Temporal Analysis with Visualization")
    print("="*60)
    
    # Grab a few test images from the dataset
    test_dir = r"data1\train\images"
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))[:3]
    
    output_dir = os.path.join("ml_results", "batch3_tests")
    os.makedirs(output_dir, exist_ok=True)
    
    if not image_paths:
        print("No images found in data1/train/images")
        return

    saved_images = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\n--- Processing: {img_name} ---")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image_rgb.copy()
        
        # Run segmentation
        masks = get_all_masks(img_path)
        if not masks:
            continue
            
        # Get depth map
        depth_map = get_depth_map(image)
        if masks[0].shape != depth_map.shape[:2]:
            depth_map = cv2.resize(depth_map, (masks[0].shape[1], masks[0].shape[0]), interpolation=cv2.INTER_LINEAR)
            
        # Matplotlib figure setup
        fig = plt.figure(figsize=(15, 6))
        ax_img = plt.subplot(1, 2, 1)
        ax_text = plt.subplot(1, 2, 2)
        
        ax_img.axis('off')
        ax_text.axis('off')
        
        text_content = f"Image: {img_name}\n" + "="*40 + "\n\n"
        
        for idx, mask in enumerate(masks):
            # Draw mask overlay
            red_layer = np.zeros_like(image_rgb)
            red_layer[:, :, 0] = 255 # Red channel in RGB
            blended = cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0)
            overlay[mask == 1] = blended[mask == 1]
            
            # Draw bounding box
            ys, xs = np.where(mask == 1)
            if len(ys) > 0 and len(xs) > 0:
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3) # Red box
                cv2.putText(overlay, f"Pothole {idx+1}", (x1, max(20, y1-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Baseline severity
            depth_features = extract_depth_features(mask, depth_map)
            if depth_features is None:
                continue
            current_severity = classify_severity(depth_features)
            
            # 1. TEMPORAL ANALYSIS
            age_estimate = estimate_pothole_age(mask, image_rgb)
            
            if age_estimate:
                progression = predict_severity_progression(
                    current_severity=current_severity,
                    age_estimate=age_estimate,
                    weather_context="freeze_thaw"
                )
                
                text_content += f"POTHOLE #{idx+1}\n"
                text_content += f"Current Severity: {progression['current_severity']} (Score: {progression['current_score']:.2f})\n"
                text_content += f"Estimated Age:    {age_estimate['age_category']} ({age_estimate['age_range']})\n"
                text_content += f"Description:      {age_estimate['age_description']}\n"
                text_content += f"Features:         Sharpness: {age_estimate['edge_sharpness']:.2f} | Crack: {age_estimate['crack_texture_score']:.2f}\n"
                
                # Try to extract DINOv2 foundation features
                try:
                    from foundation_features import extract_foundation_features, HAS_DINOV2
                    if HAS_DINOV2:
                        dino_feats = extract_foundation_features(image_rgb, mask)
                        if dino_feats:
                            text_content += f"\n-- DINOv2 Semantic Features --\n"
                            text_content += f"Dissimilarity:    {dino_feats.get('dinov2_dissimilarity', 0):.4f}\n"
                            text_content += f"Inside Variance:  {dino_feats.get('dinov2_inside_variance', 0):.4f}\n"
                except ImportError:
                    pass

                text_content += f"\n-- Deterioration Forecast (Freeze-Thaw) --\n"
                text_content += f"+30 Days:         {progression['prediction_30d']['severity']} (Score: {progression['prediction_30d']['score']:.2f})\n"
                text_content += f"+60 Days:         {progression['prediction_60d']['severity']} (Score: {progression['prediction_60d']['score']:.2f})\n"
                text_content += f"+90 Days:         {progression['prediction_90d']['severity']} (Score: {progression['prediction_90d']['score']:.2f})\n\n"
        
        ax_img.imshow(overlay)
        ax_img.set_title("Detection & Mask Overlay")
        
        # Render text
        ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes, 
                     fontsize=10, verticalalignment='top', family='monospace', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_text.set_title("Temporal Analysis & Physics Models")
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"temporal_viz_{img_name}")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_images.append(out_path)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    test_batch3_with_viz()
