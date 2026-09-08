"""
Unified Dataset Processor — Step A
====================================
Processes ALL available datasets into a single merged_dataset/ directory
with consistent YOLO segmentation label format.

Datasets handled:
  1. Kaggle Pothole Segmentation (data1/)
  2. RDD2022 country-wise ZIPs (RDD2022/)
  3. Pothole-600 stereo dataset (pothole600/)
  4. BDD100K driving images (datasets/bdd100k/) — images only, no seg labels
  5. LTPP severity data (datasets/ltpp/) — tabular only, no images
  6. KITTI Stereo Flow (datasets/kitti/data_stereo_flow/) — road scenes, no pothole labels
  7. nuScenes panoptic (datasets/nuscenes/) — metadata only, no camera images in this download

Usage:
    python scripts/process_all_datasets.py
"""

import os
import sys
import shutil
import csv
import json
import random
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: opencv-python not found. Mask-to-polygon conversion will be skipped.")

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, desc="", **kwargs):
        print(f"  {desc}...")
        return iterable

random.seed(42)

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source dirs
DATA1_DIR = PROJECT_ROOT / "data1"
RDD2022_DIR = PROJECT_ROOT / "RDD2022"
POTHOLE600_DIR = PROJECT_ROOT / "pothole600"
BDD100K_DIR = PROJECT_ROOT / "datasets" / "bdd100k"
LTPP_DIR = PROJECT_ROOT / "datasets" / "ltpp"
KITTI_DIR = PROJECT_ROOT / "datasets" / "kitti" / "data_stereo_flow"
NUSCENES_DIR = PROJECT_ROOT / "datasets" / "nuscenes"

# Output dirs
MERGED_DIR = PROJECT_ROOT / "merged_dataset"
STATS_DIR = MERGED_DIR / "dataset_stats"
REPORT_CSV = STATS_DIR / "dataset_report.csv"
SEVERITY_DIR = MERGED_DIR / "severity_labels"
STEREO_DIR = MERGED_DIR / "stereo"
RDD_TEMP_DIR = PROJECT_ROOT / "rdd_temp"

# BDD100K subsample size
BDD_SUBSAMPLE_SIZE = 500

# KITTI subsample size
KITTI_SUBSAMPLE_SIZE = 100


# --------------------------------------------------------------
# Setup
# --------------------------------------------------------------
def setup_directories():
    """Create all output directories."""
    for split in ["train", "valid", "test"]:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    SEVERITY_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        (STEREO_DIR / "disparity" / split).mkdir(parents=True, exist_ok=True)
        (STEREO_DIR / "metric_depth" / split).mkdir(parents=True, exist_ok=True)


def init_csv():
    """Initialize the dataset report CSV with headers."""
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "source", "split", "has_label",
            "image_width", "image_height", "status",
            "latitude", "longitude"
        ])


def append_to_csv(rows):
    """Append rows to the dataset report CSV."""
    if not rows:
        return
    with open(REPORT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# --------------------------------------------------------------
# 1. Kaggle Pothole Segmentation (data1/)
# --------------------------------------------------------------
def process_kaggle():
    """Copy Kaggle pothole segmentation data from data1/."""
    print("\n" + "=" * 60)
    print("1. Processing Kaggle Pothole Segmentation (data1/)")
    print("=" * 60)

    if not DATA1_DIR.exists():
        print("  SKIP: data1/ directory not found.")
        return 0

    csv_rows = []
    total = 0

    for split in ["train", "valid"]:
        img_dir = DATA1_DIR / split / "images"
        lbl_dir = DATA1_DIR / split / "labels"

        if not img_dir.exists():
            print(f"  SKIP: {img_dir} not found.")
            continue

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        for img_path in tqdm(images, desc=f"Kaggle {split}"):
            new_name = f"kaggle_{img_path.name}"
            new_img_dest = MERGED_DIR / split / "images" / new_name
            shutil.copy2(img_path, new_img_dest)

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            has_label = False
            if lbl_path.exists():
                new_lbl_dest = MERGED_DIR / split / "labels" / f"kaggle_{lbl_path.name}"
                shutil.copy2(lbl_path, new_lbl_dest)
                has_label = True

            csv_rows.append([new_name, "kaggle", split, has_label, "", "", "ok", "", ""])
            total += 1

    append_to_csv(csv_rows)
    print(f"  Total: {total} images processed.")
    return total


# --------------------------------------------------------------
# 2. RDD2022 (country-wise ZIPs)
# --------------------------------------------------------------
def process_rdd2022():
    """Extract and process RDD2022 country ZIPs, filtering for D40 (pothole) annotations."""
    print("\n" + "=" * 60)
    print("2. Processing RDD2022 (country-wise ZIPs)")
    print("=" * 60)

    if not RDD2022_DIR.exists():
        print("  SKIP: RDD2022/ directory not found.")
        return 0

    found_zips = sorted(RDD2022_DIR.glob("*.zip"))
    if not found_zips:
        print("  SKIP: No ZIP files found in RDD2022/.")
        return 0

    print(f"  Found {len(found_zips)} ZIP files: {[z.stem for z in found_zips]}")
    RDD_TEMP_DIR.mkdir(exist_ok=True)
    total = 0

    for zip_path in found_zips:
        country = zip_path.stem
        extract_dir = RDD_TEMP_DIR / country

        # Extract if not already done
        if not extract_dir.exists():
            print(f"  Extracting {country}...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                print(f"  ERROR: {zip_path.name} is corrupted. Skipping.")
                continue

        # Find all XML annotation files
        xml_files = list(extract_dir.rglob("*.xml"))
        print(f"  {country}: Found {len(xml_files)} XML files.")

        valid_pairs = []
        for xml_path in tqdm(xml_files, desc=f"Filtering {country}"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                has_d40 = False
                pothole_boxes = []
                size_elem = root.find("size")
                if size_elem is None:
                    continue
                w_img = int(size_elem.find("width").text)
                h_img = int(size_elem.find("height").text)
                if w_img == 0 or h_img == 0:
                    continue

                for obj in root.findall("object"):
                    name = obj.find("name").text
                    if name == "D40":
                        has_d40 = True
                        bndbox = obj.find("bndbox")
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)

                        # Convert bbox to 4-point polygon (YOLO seg format)
                        x1 = max(0.0, min(1.0, xmin / w_img))
                        y1 = max(0.0, min(1.0, ymin / h_img))
                        x2 = max(0.0, min(1.0, xmax / w_img))
                        y2 = max(0.0, min(1.0, ymin / h_img))
                        x3 = max(0.0, min(1.0, xmax / w_img))
                        y3 = max(0.0, min(1.0, ymax / h_img))
                        x4 = max(0.0, min(1.0, xmin / w_img))
                        y4 = max(0.0, min(1.0, ymax / h_img))

                        pothole_boxes.append(
                            f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                            f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                        )

                if has_d40:
                    img_name = root.find("filename").text
                    # Try multiple locations for the image
                    img_path = xml_path.parent / img_name
                    if not img_path.exists():
                        img_path = xml_path.parent.parent / "images" / img_name
                    if not img_path.exists():
                        img_path = xml_path.parent.parent.parent / "images" / img_name

                    if img_path.exists():
                        valid_pairs.append({
                            "xml": xml_path,
                            "img": img_path,
                            "labels": pothole_boxes,
                            "w": w_img,
                            "h": h_img,
                        })
            except Exception:
                continue

        # Split 80/10/10
        random.shuffle(valid_pairs)
        n = len(valid_pairs)
        train_idx = int(0.8 * n)
        val_idx = int(0.9 * n)

        csv_rows = []
        country_lower = country.lower()
        for i, pair in enumerate(tqdm(valid_pairs, desc=f"Writing {country}")):
            if i < train_idx:
                split = "train"
            elif i < val_idx:
                split = "valid"
            else:
                split = "test"

            orig_name = pair["img"].name
            new_name = f"rdd_{country_lower}_{orig_name}"
            new_img_dest = MERGED_DIR / split / "images" / new_name
            new_lbl_dest = MERGED_DIR / split / "labels" / f"{Path(new_name).stem}.txt"

            shutil.copy2(pair["img"], new_img_dest)
            with open(new_lbl_dest, "w", encoding="utf-8") as f:
                f.write("\n".join(pair["labels"]))

            csv_rows.append([
                new_name, f"rdd2022_{country_lower}", split,
                True, pair["w"], pair["h"], "ok", "", ""
            ])
            total += 1

        append_to_csv(csv_rows)
        print(f"  {country}: {len(valid_pairs)} D40 images processed.")

    print(f"  Total RDD2022: {total} images.")
    return total


# --------------------------------------------------------------
# 3. Pothole-600
# --------------------------------------------------------------
def process_masks_to_yolo(mask_path, w, h):
    """Convert a binary mask image to YOLO polygon format."""
    if not HAS_CV2:
        return []
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) == 1:
            cnt = cnt.reshape(-1, 2)

        coords = []
        for point in cnt:
            x, y = point[0] / w, point[1] / h
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            coords.append(f"{x:.6f} {y:.6f}")

        if len(coords) >= 3:
            polygons.append("0 " + " ".join(coords))
    return polygons


def process_disparity(disp_path, dest_disp_path, dest_depth_path, focal_length=1.0, baseline=1.0):
    """Convert disparity image to numpy arrays."""
    if not HAS_CV2:
        return False
    disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
    if disp is None:
        return False

    disp_float = disp.astype(np.float32)
    depth_meters = (focal_length * baseline) / (disp_float + 1e-8)

    np.save(str(dest_disp_path), disp_float)
    np.save(str(dest_depth_path), depth_meters)
    return True


def process_pothole600():
    """Process Pothole-600 dataset with its training/testing/validation splits."""
    print("\n" + "=" * 60)
    print("3. Processing Pothole-600 (pothole600/)")
    print("=" * 60)

    if not POTHOLE600_DIR.exists():
        print("  SKIP: pothole600/ directory not found.")
        return 0

    # Map Pothole-600 splits to our splits
    split_mapping = {
        "training": "train",
        "validation": "valid",
        "testing": "test",
    }

    csv_rows = []
    severity_rows = []
    total = 0

    for p600_split, our_split in split_mapping.items():
        split_dir = POTHOLE600_DIR / p600_split
        if not split_dir.exists():
            print(f"  SKIP: {split_dir} not found.")
            continue

        rgb_dir = split_dir / "rgb"
        label_dir = split_dir / "label"
        tdisp_dir = split_dir / "tdisp"

        if not rgb_dir.exists():
            print(f"  SKIP: {rgb_dir} not found.")
            continue

        images = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
        print(f"  {p600_split}: Found {len(images)} images.")

        for img_path in tqdm(images, desc=f"Pothole-600 {p600_split}"):
            img = cv2.imread(str(img_path)) if HAS_CV2 else None
            if img is None and HAS_CV2:
                continue

            if HAS_CV2:
                h, w = img.shape[:2]
            else:
                h, w = 0, 0

            stem = img_path.stem
            new_name = f"p600_{img_path.name}"
            new_stem = Path(new_name).stem

            # Copy image
            shutil.copy2(img_path, MERGED_DIR / our_split / "images" / new_name)

            # Convert mask to YOLO polygon
            mask_path = label_dir / f"{stem}.png"
            polygons = []
            if mask_path.exists() and HAS_CV2:
                polygons = process_masks_to_yolo(mask_path, w, h)

            if polygons:
                lbl_dest = MERGED_DIR / our_split / "labels" / f"{new_stem}.txt"
                with open(lbl_dest, "w") as f:
                    f.write("\n".join(polygons))

            # Process disparity if available
            tdisp_path = tdisp_dir / f"{stem}.png"
            if tdisp_path.exists():
                disp_dest = STEREO_DIR / "disparity" / our_split / f"{new_stem}.npy"
                depth_dest = STEREO_DIR / "metric_depth" / our_split / f"{new_stem}.npy"
                process_disparity(tdisp_path, disp_dest, depth_dest)

            has_label = len(polygons) > 0
            csv_rows.append([new_name, "pothole600", our_split, has_label, w, h, "ok", "", ""])
            total += 1

    append_to_csv(csv_rows)

    # Write severity labels CSV (header only for now — Pothole-600 doesn't have text severity)
    sev_csv_path = SEVERITY_DIR / "pothole600_annotations.csv"
    with open(sev_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "severity_int", "severity_label", "has_disparity"])
        writer.writerows(severity_rows)

    print(f"  Total Pothole-600: {total} images.")
    return total


# --------------------------------------------------------------
# 4. BDD100K (images only — no seg labels from det_20 format)
# --------------------------------------------------------------
def process_bdd100k():
    """
    Process BDD100K images for weather/lighting diversity.

    The det_20 labels are rendered as JPG images (not JSON), so we cannot
    extract YOLO polygon labels. We subsample ~500 images for diversity.
    """
    print("\n" + "=" * 60)
    print("4. Processing BDD100K (images only, no seg labels)")
    print("=" * 60)

    # Actual path: datasets/bdd100k/images/bdd100k_images_100k/100k/{train,val,test}
    bdd_images_root = BDD100K_DIR / "images" / "bdd100k_images_100k" / "100k"

    if not bdd_images_root.exists():
        print(f"  SKIP: {bdd_images_root} not found.")
        return 0

    csv_rows = []
    total = 0

    # Map BDD splits to our splits
    bdd_split_map = {"train": "train", "val": "valid", "test": "test"}

    for bdd_split, our_split in bdd_split_map.items():
        split_dir = bdd_images_root / bdd_split
        if not split_dir.exists():
            print(f"  SKIP: {split_dir} not found.")
            continue

        images = sorted(split_dir.glob("*.jpg"))
        print(f"  {bdd_split}: Found {len(images)} images.")

        # Subsample for manageable size
        per_split = BDD_SUBSAMPLE_SIZE
        if bdd_split == "val":
            per_split = max(50, BDD_SUBSAMPLE_SIZE // 5)
        elif bdd_split == "test":
            per_split = max(50, BDD_SUBSAMPLE_SIZE // 5)

        if len(images) > per_split:
            sampled = random.sample(images, per_split)
        else:
            sampled = images

        for img_path in tqdm(sampled, desc=f"BDD100K {bdd_split}"):
            new_name = f"bdd_{img_path.name}"
            new_img_dest = MERGED_DIR / our_split / "images" / new_name
            shutil.copy2(img_path, new_img_dest)

            # No labels available from det_20 format
            csv_rows.append([new_name, "bdd100k", our_split, False, "", "", "ok", "", ""])
            total += 1

    append_to_csv(csv_rows)
    print(f"  Total BDD100K: {total} images (subsampled, no labels).")
    return total


# --------------------------------------------------------------
# 5. LTPP (tabular severity data only)
# --------------------------------------------------------------
def process_ltpp():
    """
    Process LTPP data — convert xlsx sheets to CSVs.
    No images to add to merged_dataset, only severity metadata.
    """
    print("\n" + "=" * 60)
    print("5. Processing LTPP (tabular severity data)")
    print("=" * 60)

    xlsx_path = LTPP_DIR / "raw" / "Bucket_145337.xlsx"
    if not xlsx_path.exists():
        print(f"  SKIP: {xlsx_path} not found.")
        return 0

    # Run the dedicated LTPP converter
    try:
        from convert_ltpp_xlsx import main as ltpp_main
        # If imported from scripts dir, run it
        ltpp_main()
    except ImportError:
        # Try running as subprocess
        import subprocess
        converter_path = PROJECT_ROOT / "scripts" / "convert_ltpp_xlsx.py"
        if converter_path.exists():
            print("  Running convert_ltpp_xlsx.py as subprocess...")
            result = subprocess.run(
                [sys.executable, str(converter_path)],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
        else:
            print("  SKIP: convert_ltpp_xlsx.py not found.")

    # Also copy LTPP severity data to merged_dataset severity_labels
    ltpp_sev = LTPP_DIR / "processed" / "ltpp_pothole_severity.csv"
    if ltpp_sev.exists():
        dest = SEVERITY_DIR / "ltpp_severity.csv"
        shutil.copy2(ltpp_sev, dest)
        # Count records
        with open(ltpp_sev, "r") as f:
            count = max(0, sum(1 for _ in f) - 1)
        print(f"  Copied LTPP severity data ({count} records) to merged_dataset/severity_labels/")
        return count
    return 0


# --------------------------------------------------------------
# 6. KITTI Stereo Flow (road scenes, no pothole labels)
# --------------------------------------------------------------
def process_kitti():
    """
    Process KITTI stereo flow images. Uses colored_0 (left color images)
    since image_2 is empty in this download.
    """
    print("\n" + "=" * 60)
    print("6. Processing KITTI Stereo Flow (road scenes, no labels)")
    print("=" * 60)

    kitti_train = KITTI_DIR / "training"
    kitti_test = KITTI_DIR / "testing"

    if not kitti_train.exists():
        print(f"  SKIP: {kitti_train} not found.")
        return 0

    # Use colored_0 (left color images) since image_2 is empty
    img_dir = kitti_train / "colored_0"
    if not img_dir.exists():
        print(f"  SKIP: {img_dir} not found.")
        return 0

    # Get all images (colored_0 has pairs like 000000_10.png, 000000_11.png)
    # Take only the _10 frames (first of pair)
    all_images = sorted(img_dir.glob("*_10.png"))
    print(f"  Found {len(all_images)} left-frame images in colored_0.")

    # Subsample
    if len(all_images) > KITTI_SUBSAMPLE_SIZE:
        sampled = random.sample(all_images, KITTI_SUBSAMPLE_SIZE)
    else:
        sampled = all_images

    # Split 80/10/10
    random.shuffle(sampled)
    n = len(sampled)
    train_idx = int(0.8 * n)
    val_idx = int(0.9 * n)

    csv_rows = []
    total = 0
    disp_noc_dir = kitti_train / "disp_noc"

    for i, img_path in enumerate(tqdm(sampled, desc="KITTI")):
        if i < train_idx:
            split = "train"
        elif i < val_idx:
            split = "valid"
        else:
            split = "test"

        new_name = f"kitti_{img_path.name}"
        new_img_dest = MERGED_DIR / split / "images" / new_name
        shutil.copy2(img_path, new_img_dest)

        # Copy disparity ground truth if available
        disp_path = disp_noc_dir / img_path.name
        if disp_path.exists():
            new_stem = Path(new_name).stem
            disp_dest = STEREO_DIR / "disparity" / split / f"{new_stem}.npy"
            depth_dest = STEREO_DIR / "metric_depth" / split / f"{new_stem}.npy"
            process_disparity(disp_path, disp_dest, depth_dest,
                              focal_length=721.5377, baseline=0.5372)

        csv_rows.append([new_name, "kitti_stereo", split, False, "", "", "ok", "", ""])
        total += 1

    append_to_csv(csv_rows)
    print(f"  Total KITTI: {total} images (subsampled, no pothole labels).")
    return total


# --------------------------------------------------------------
# 7. nuScenes (metadata only)
# --------------------------------------------------------------
def process_nuscenes():
    """
    Process nuScenes metadata. This download only contains panoptic
    annotations (npz files) and metadata JSONs, not camera images.
    We extract the metadata for future use.
    """
    print("\n" + "=" * 60)
    print("7. Processing nuScenes (metadata only)")
    print("=" * 60)

    nuscenes_raw = NUSCENES_DIR / "raw"
    panoptic_root = nuscenes_raw / "nuScenes-panoptic-v1.0-all"

    if not panoptic_root.exists():
        print(f"  SKIP: {panoptic_root} not found.")
        return 0

    # Check for camera images (samples/ directory)
    samples_dir = panoptic_root / "samples"
    has_images = samples_dir.exists()

    if not has_images:
        print("  INFO: No camera images found (samples/ directory missing).")
        print("  This download only contains panoptic annotation masks + metadata.")
        print("  Extracting metadata for future use when full dataset is downloaded.")

    # Process metadata JSONs
    metadata_count = 0
    nuscenes_metadata_dir = MERGED_DIR / "dataset_stats" / "nuscenes_metadata"
    nuscenes_metadata_dir.mkdir(parents=True, exist_ok=True)

    for version_dir in ["v1.0-mini", "v1.0-trainval", "v1.0-test"]:
        meta_dir = panoptic_root / version_dir
        if not meta_dir.exists():
            continue

        json_files = list(meta_dir.glob("*.json"))
        for json_file in json_files:
            dest = nuscenes_metadata_dir / f"{version_dir}_{json_file.name}"
            shutil.copy2(json_file, dest)
            metadata_count += 1

    # Count panoptic masks
    panoptic_dir = panoptic_root / "panoptic"
    npz_count = 0
    if panoptic_dir.exists():
        for sub in panoptic_dir.iterdir():
            if sub.is_dir():
                npz_count += len(list(sub.glob("*.npz")))

    print(f"  Metadata files copied: {metadata_count}")
    print(f"  Panoptic masks available: {npz_count}")
    print(f"  Camera images: {'Available' if has_images else 'NOT available in this download'}")

    if not has_images:
        print("\n  To add nuScenes images in the future:")
        print("  1. Download the full v1.0-mini or v1.0-trainval from nuscenes.org")
        print("  2. Place the 'samples/' directory inside nuScenes-panoptic-v1.0-all/")
        print("  3. Re-run this script")

    return 0  # No images added


# --------------------------------------------------------------
# Summary
# --------------------------------------------------------------
def generate_final_summary():
    """Generate the final dataset.yaml and summary statistics."""
    print("\n" + "=" * 60)
    print("Generating Final Summary")
    print("=" * 60)

    total_counts = {"train": 0, "valid": 0, "test": 0}
    source_counts = {}

    for split in ["train", "valid", "test"]:
        img_dir = MERGED_DIR / split / "images"
        if not img_dir.exists():
            continue
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        total_counts[split] = len(images)

        for img_path in images:
            name = img_path.name
            if name.startswith("kaggle_"):
                src = "kaggle"
            elif name.startswith("rdd_"):
                # Extract country: rdd_<country>_<rest>
                parts = name.split("_", 2)
                src = f"rdd_{parts[1]}" if len(parts) > 2 else "rdd_unknown"
            elif name.startswith("p600_"):
                src = "pothole600"
            elif name.startswith("bdd_"):
                src = "bdd100k"
            elif name.startswith("kitti_"):
                src = "kitti"
            else:
                src = "unknown"

            if src not in source_counts:
                source_counts[src] = {"train": 0, "valid": 0, "test": 0}
            source_counts[src][split] += 1

    # Count labeled vs unlabeled
    labeled = 0
    unlabeled = 0
    for split in ["train", "valid", "test"]:
        img_dir = MERGED_DIR / split / "images"
        lbl_dir = MERGED_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                labeled += 1
            else:
                unlabeled += 1

    # Generate dataset.yaml
    yaml_path = MERGED_DIR / "dataset.yaml"
    grand_total = sum(total_counts.values())
    yaml_content = f"""path: merged_dataset
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['pothole']

# Dataset statistics (auto-generated)
# train_images: {total_counts['train']}
# valid_images: {total_counts['valid']}
# test_images: {total_counts['test']}
# total_images: {grand_total}
# labeled_images: {labeled}
# unlabeled_images: {unlabeled}
# sources: {', '.join(sorted(source_counts.keys()))}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Print summary
    print(f"\n  Total images: {grand_total}")
    print(f"    Train: {total_counts['train']}")
    print(f"    Valid: {total_counts['valid']}")
    print(f"    Test:  {total_counts['test']}")
    print(f"  Labeled (with YOLO seg): {labeled}")
    print(f"  Unlabeled (images only):  {unlabeled}")
    print()

    # Source table
    print("  +---------------------+--------+-------+------+--------+")
    print("  | Source              | Train  | Valid | Test | Total  |")
    print("  +---------------------+--------+-------+------+--------+")
    for src in sorted(source_counts.keys()):
        c = source_counts[src]
        tot = c["train"] + c["valid"] + c["test"]
        print(f"  | {src:<19} | {c['train']:>6} | {c['valid']:>5} | {c['test']:>4} | {tot:>6} |")
    print("  +---------------------+--------+-------+------+--------+")
    print(f"  | TOTAL               | {total_counts['train']:>6} | {total_counts['valid']:>5} | {total_counts['test']:>4} | {grand_total:>6} |")
    print("  +---------------------+--------+-------+------+--------+")

    # Write summary to file
    summary_path = STATS_DIR / "final_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset Processing Summary -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images: {grand_total}\n")
        f.write(f"  Train: {total_counts['train']}\n")
        f.write(f"  Valid: {total_counts['valid']}\n")
        f.write(f"  Test:  {total_counts['test']}\n")
        f.write(f"Labeled: {labeled}\n")
        f.write(f"Unlabeled: {unlabeled}\n\n")
        f.write("Per-source breakdown:\n")
        for src in sorted(source_counts.keys()):
            c = source_counts[src]
            tot = c["train"] + c["valid"] + c["test"]
            f.write(f"  {src}: {tot} (train={c['train']}, valid={c['valid']}, test={c['test']})\n")

    print(f"\n  Summary written to: {summary_path}")
    print(f"  Dataset YAML written to: {yaml_path}")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    print("+============================================================+")
    print("|    RoadLens -- Unified Dataset Processor (Step A)           |")
    print("+============================================================+")
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Output: {MERGED_DIR}")

    # Setup
    setup_directories()
    init_csv()

    # Process each dataset
    results = {}
    results["kaggle"] = process_kaggle()
    results["rdd2022"] = process_rdd2022()
    results["pothole600"] = process_pothole600()
    results["bdd100k"] = process_bdd100k()
    results["ltpp"] = process_ltpp()
    results["kitti"] = process_kitti()
    results["nuscenes"] = process_nuscenes()

    # Generate summary
    generate_final_summary()

    # Final report
    print("\n" + "+============================================================+")
    print("|    Processing Complete!                                    |")
    print("+============================================================+")
    for name, count in results.items():
        status = f"{count} items" if count > 0 else "skipped/metadata only"
        print(f"  {name:<15} -> {status}")


if __name__ == "__main__":
    main()
