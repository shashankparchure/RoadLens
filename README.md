<div align="center">
  <img src="docs/assets/hero.svg" alt="RoadLens Hero" width="800"/>

  <h1>🛣️ RoadLens: Advanced Vision-Based Pothole Detection</h1>

  <p>
    <em>A next-generation road hazard analysis stack powered by YOLOv8, Depth-Anything-V2, DINOv2, and Shape-From-Shading.</em>
  </p>

  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white&style=for-the-badge)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black&style=for-the-badge)](https://react.dev/)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-111111?style=for-the-badge)](https://docs.ultralytics.com/)
</div>

<hr/>

## ✨ What Makes RoadLens Different?

RoadLens goes far beyond drawing basic bounding boxes around potholes. It builds a comprehensive 3D understanding of the road surface using multi-modal AI pipelines:

* 🌊 **Water Hazard Detection**: Algorithms detect water-filled potholes which pose significantly higher risks to drivers.
* 🕳️ **Pseudo-Depth Profiling**: Combines neural depth networks with photometric **Shape-From-Shading (SFS)** to capture the jagged micro-textures of deep craters.
* 🧠 **Semantic Intelligence (DINOv2)**: Differentiates between actual structural road damage and deceptive flat shadows cast by trees or vehicles.
* ⏱️ **Temporal Degradation Tracking**: Tracks pothole evolution across sequential frames to predict degradation rates over time.

---

## 🏗️ System Architecture

![Pipeline Overview](docs/assets/workflow.svg)

The following diagram illustrates the multi-modal data flow from image upload to final severity classification:

```mermaid
graph TD
    A[Upload Road Image] --> B{YOLOv8 Segmentation}
    
    %% Parallel Feature Extraction Paths
    B -->|Mask Generation| C[Feature Extractor]
    A -->|Monocular Processing| D[Depth Anything V2]
    A -->|Photometric Analysis| E[Shape-From-Shading]
    A -->|Semantic Analysis| F[DINOv2 Foundation Features]
    
    %% Merging Features
    D -->|Global Depth Map| C
    E -->|Micro-texture Depth| C
    F -->|Semantic Tensors| C
    
    %% Water Detection Branch
    A -->|Color & Reflection| W[Water Hazard Detector]
    W -->|Water Presence Boolean| C
    
    %% Classification
    C -->|Feature Vector: Volume, Depth Variance, Context| G[Machine Learning Ensemble]
    C -->|Fallback Constraints| H[Rule-Based Classifier]
    
    %% Output
    G --> I((Final Severity Score & Alert JSON))
    H --> I
    
    style A fill:#2d3436,stroke:#74b9ff,stroke-width:2px,color:#fff
    style I fill:#00b894,stroke:#55efc4,stroke-width:4px,color:#fff
    style W fill:#0984e3,stroke:#74b9ff,color:#fff
```

---

## 📊 Visualizing Results

The system is equipped with an interactive **React Bento-Dashboard** that visualizes the results. Internally, the pipeline generates robustness matrices and performance charts.

*(Check the `ml_results/` directory for generated visualizations like the one below, which compares model accuracy across different condition scenarios!)*

> <img src="ml_results/robustness_comparison.png" alt="Robustness Comparison" width="600" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>

---

## 📁 Directory Architecture

```text
RoadLens/
├── api.py                        # FastAPI backend entry point
├── main.py                       # CLI rule-based pipeline
├── inference.py                  # CLI ML-based inference pipeline
├── src/                          # Core & Advanced Algorithms
│   ├── core/                     # YOLO, Features, ML Classifiers
│   └── advanced/                 # SFS, Water Detection, DINOv2
├── tests/                        # Suite of testing and verification scripts
├── ml_models/                    # Trained ML artifacts
├── ml_results/                   # Evaluation results, charts, and metrics
├── scripts/                      # Utility scripts (dataset conversion, etc.)
└── web-ui/                       # React/Vite Frontend Dashboard
```

---

## 🚀 Quick Start Guide

<details>
<summary><strong>1. Environment Setup (Click to expand)</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```
</details>

<details>
<summary><strong>2. External Weights & Models</strong></summary>

You must download the foundational models to run local inference:
1. **Depth-Anything-V2**: `git clone https://github.com/DepthAnything/Depth-Anything-V2.git`
2. **Depth Checkpoint**: Place `depth_anything_v2_vits.pth` in `Depth-Anything-V2/checkpoints/`
3. **YOLO Weights**: Place your trained `best.pt` in `yolo-segmentation/model/`
</details>

<details>
<summary><strong>3. Running the Stack</strong></summary>

**Backend:**
```powershell
python api.py
```

**Frontend:**
```powershell
cd web-ui
npm install
npm run dev
```
</details>

---

## 🧪 Testing & Validation

We maintain a rigorous test suite to ensure the pipeline handles diverse road conditions correctly.

Run the end-to-end integration test:
```powershell
python tests/test_pipeline.py
```

Run specific module verifications:
```powershell
python tests/test_water_detection.py
python tests/test_depth_api.py
```

*Results, logs, and visualized outputs from these tests are automatically saved into the `ml_results/` folder for immediate inspection.*
