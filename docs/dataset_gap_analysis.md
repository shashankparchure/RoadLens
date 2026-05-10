# Dataset Gap Analysis

## Current Dataset Coverage

| Source | Images | Annotations | Depth Maps | Weather | Night | Water | Real Severity Labels |
|--------|--------|-------------|------------|---------|-------|-------|---------------------|
| data1 (Kaggle) | 665 | YOLO segmentation | ✅ Pre-computed | Clear only | ❌ | ❌ | ❌ Pseudo-labels |
| Pothole-600 | ~600 | Bounding box | ❌ | Clear only | ❌ | ❌ | ❌ CSV empty |
| RDD2022 | ~11K | Bounding box (D40) | ❌ | Clear mostly | Rare | ❌ | ❌ Class only |
| merged_dataset | ~1,700 | YOLO segmentation | ✅ Computed | Clear only | ❌ | ❌ | ❌ Pseudo-labels |

**Total unique pothole images: ~2,300**

## Identified Gaps

### Critical Gaps

1. **Zero adverse weather images (rain, fog, snow)**
   - All training images are clear weather / daytime
   - System has never seen real rain, fog, or low-light conditions
   - Impact: Depth model predictions are untested under degraded visibility
   - Mitigation: `adverse_conditions.py` provides synthetic augmentation

2. **Zero real severity ground truth labels**
   - All severity labels are pseudo-labels derived from depth statistics
   - Pothole-600 CSV was expected to have severity ratings but is empty
   - Impact: Classifier validation is circular (features derive labels, labels train classifier)
   - Mitigation: LTPP database or custom IMU collection would provide real labels

3. **Zero water-filled pothole annotations**
   - No images are labeled as water-filled vs dry
   - Water detection module was tuned on 4 hand-selected test images
   - Impact: Water detection thresholds may not generalize
   - Mitigation: SWD dataset or manual annotation of existing images

### Moderate Gaps

4. **No nighttime images**
   - Headlight illumination creates different shadow patterns
   - Street lighting produces sodium vapor color shift
   - Impact: Color-based features (saturation, brightness) may behave differently

5. **No illumination metadata**
   - SfS reconstruction assumes illumination direction from image gradients
   - No ground truth light direction for validation
   - Impact: SfS accuracy is unverified against known illumination

6. **Limited geographic diversity**
   - Primarily Indian and Japanese road surfaces
   - Missing: European cobblestone, US highways, African unpaved roads
   - Impact: Texture-based features may not transfer across surface types

## Recommended Datasets to Fill Gaps

| Priority | Dataset | Fills Gap | Effort | URL |
|----------|---------|-----------|--------|-----|
| 🔴 High | **BDD100K** | Weather diversity (rain/fog/night) | Download + filter | bdd-data.berkeley.edu |
| 🔴 High | **LTPP (FHWA)** | Real severity labels | API access + processing | infopave.fhwa.dot.gov |
| 🟡 Medium | **SWD (Chen et al.)** | Water-filled annotations | Paper request | Various publications |
| 🟡 Medium | **nuScenes** | Night + rain + IMU/GPS | Large download (~1TB) | nuscenes.org |
| 🟢 Low | **KITTI Stereo** | Depth ground truth for SfS validation | Small download | kitti.is.tue.mpg.de |
| 🟢 Low | **Mapillary Vistas** | Geographic diversity (66 countries) | API access | mapillary.com/dataset |

## Augmentation Strategy (Immediate Mitigation)

Since downloading and processing external datasets requires significant effort, the immediate strategy is **synthetic augmentation** using `adverse_conditions.py`:

1. For each training image, generate 1 random adverse variant (rain/fog/night/shadow)
2. Re-extract features on the augmented image
3. Add augmented sample with the same severity label
4. Training set approximately doubles in size

This is controlled by `USE_ADVERSE_AUGMENTATION = True` in `ml_classifier.py`.

**Expected impact**: The augmented models should generalize better to real-world conditions because they've been trained on both clean and degraded inputs. Geometry features should maintain accuracy while depth features may slightly degrade, further validating the geometry-based approach.
