# RoadLens: Advanced Vision-Based Pothole Detection

![Project Hero](docs/assets/hero.svg)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-111111)](https://docs.ultralytics.com/)

RoadLens is an advanced pothole analysis stack that goes beyond basic bounding boxes. It combines YOLO-based segmentation, multi-modal depth estimation (including Shape-From-Shading), rule-based logic, DINOv2 foundation features, and advanced hazard tracking in a single robust workflow.

## Table of Contents
- [1. What This Project Does](#1-what-this-project-does)
- [2. Key Features](#2-key-features)
- [3. Current Directory Structure](#3-current-directory-structure)
- [4. How It Works](#4-how-it-works)
- [5. First-Time Setup](#5-first-time-setup)
- [6. Run the Project](#6-run-the-project)
- [7. Testing and Evaluation](#7-testing-and-evaluation)

## 1. What This Project Does

This project analyzes road images and estimates pothole severity and road condition hazards by combining:
- YOLO-based pothole segmentation
- Monocular depth estimation augmented with **Shape-From-Shading (SFS)**
- **Water Hazard Detection** to identify hidden or submerged potholes
- **Temporal Analysis** to track road degradation over time
- **Foundation Features (DINOv2)** for robust semantic cross-validation in adverse conditions
- Machine learning ensemble predictions
- An interactive React dashboard (Bento-style) for real-time visualization and insights

## 2. Key Features

- **Robust Depth Profiling**: Uses a hybrid approach combining neural depth estimation with photometric Shape-From-Shading to capture jagged micro-textures of craters.
- **Environmental Hazard Awareness**: Detects water-filled potholes which pose significantly higher risks, dynamically adjusting severity scores.
- **Semantic Intelligence**: Integrates DINOv2 to distinguish between actual structural road damage and deceptive flat shadows.
- **Temporal Tracking**: Tracks pothole evolution across sequential frames to predict degradation rates.

## 3. Current Directory Structure

```text
RoadLens/
|-- README.md
|-- api.py                        # FastAPI backend entry point
|-- main.py                       # CLI rule-based pipeline
|-- inference.py                  # CLI ML-based inference pipeline
|-- classifier.py                 # Rule-based severity classifier
|-- ml_classifier.py              # ML model trainer and evaluator
|-- segmentation.py               # YOLOv8 inference
|-- features.py                   # Core feature extraction
|-- foundation_features.py        # DINOv2 semantic feature extraction
|-- shape_from_shading.py         # Photometric pseudo-depth estimation
|-- water_detection.py            # Water hazard analysis
|-- temporal_analysis.py          # Frame-to-frame degradation tracking
|-- adverse_conditions.py         # Weather and lighting compensation
|-- road_segment_analysis.py      # Spatial context analysis
|-- tests/                        # Suite of testing and verification scripts
|-- ml_models/                    # Trained ML artifacts
|-- ml_results/                   # Evaluation results, charts, and metrics
|-- docs/                         # Project documentation and plans
|-- scripts/                      # Utility scripts (dataset conversion, etc.)
`-- web-ui/                       # React/Vite Frontend Application
    |-- src/
    |   |-- App.jsx
    |   `-- components/
    |       `-- detection/        # Specialized React UI components
```

## 4. How It Works

### End-to-End Flow

1. **Upload**: Road image is uploaded via the React Web UI or API.
2. **Segmentation**: YOLOv8 extracts precise pixel masks of potholes.
3. **Multi-Modal Feature Extraction**:
   - Depth Anything V2 estimates global scene depth.
   - Shape-From-Shading estimates localized crater textures.
   - DINOv2 extracts semantic features to rule out shadow-illusions.
   - Water detection algorithms analyze reflections/color to spot hidden hazards.
4. **Classification**: Extracted features (volume, depth variance, water presence, context) are fed into trained ML ensembles and rule-based fallback systems.
5. **Output**: System yields a packaged JSON with severity, hazard alerts, and temporal prognosis.

## 5. First-Time Setup

### 5.1 Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 5.2 Backend Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

### 5.3 External Assets You Need
1. **Depth-Anything-V2 repository**: `git clone https://github.com/DepthAnything/Depth-Anything-V2.git`
2. **Depth checkpoint**: Place `depth_anything_v2_vits.pth` in `Depth-Anything-V2/checkpoints/`
3. **YOLO weights**: Place your trained `best.pt` in `yolo-segmentation/model/`

### 5.4 Frontend Setup
```powershell
cd web-ui
copy .env.example .env
npm install
```
Ensure `VITE_API_BASE_URL=http://localhost:8000` is set in `web-ui/.env`.

## 6. Run the Project

### Start backend API
```powershell
python api.py
```
Health check: `http://localhost:8000/healthz`

### Start frontend
```powershell
cd web-ui
npm run dev
```

### Run CLI inference
```powershell
python inference.py <path_to_image> --output_dir output --no_show
```

## 7. Testing and Evaluation

All test scripts have been consolidated into the `tests/` directory.

- **Run full pipeline tests**:
  ```powershell
  python tests/test_pipeline.py
  ```
- **Test specific modules (Water, API, Batch processing)**:
  ```powershell
  python tests/test_water_detection.py
  python tests/test_depth_api.py
  python tests/test_batch3.py
  ```

Results from tests and model training are automatically saved to `ml_results/` for inspection.
