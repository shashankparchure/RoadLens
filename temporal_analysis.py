"""
Temporal Analysis Module
========================
Estimates pothole age and predicts severity progression over time
using physics-informed deterioration models.

Age Estimation:
    Uses curvature sharpness, boundary regularity, surrounding crack
    texture (Gabor filters), and depth-area ratio to classify potholes
    as Fresh (<2 weeks), Developing (2-8 weeks), Established (2-6 months),
    or Chronic (>6 months).

Severity Progression:
    Physics-based model of pavement deterioration.  Predicts how a
    pothole's severity will change over 30/60/90 day windows based on
    current condition, estimated age, and climate context.

Usage:
    from temporal_analysis import estimate_pothole_age, predict_severity_progression
    age = estimate_pothole_age(mask, image_rgb, curvature_features)
    progression = predict_severity_progression(features, age)
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Age Estimation
# ═══════════════════════════════════════════════════════════════════════════

# Age categories with physical interpretation
AGE_CATEGORIES = {
    "Fresh":       {"range": "<2 weeks",   "description": "Recent formation, sharp edges, clean break"},
    "Developing":  {"range": "2-8 weeks",  "description": "Edges softening, minor secondary cracking"},
    "Established": {"range": "2-6 months", "description": "Rounded edges, surrounding cracks, debris"},
    "Chronic":     {"range": ">6 months",  "description": "Heavily eroded, complex shape, extensive cracking"},
}


def _compute_edge_sharpness(curvature_features: Dict[str, Any]) -> float:
    """
    Compute edge sharpness from curvature features.

    Fresh potholes have very sharp, angular breaks (high max_curvature).
    Older potholes have been rounded by traffic and weather (low curvature).

    Returns a value in [0, 1] where 1 = extremely sharp (fresh).
    """
    max_curv = float(curvature_features.get("max_curvature", 0))
    p90_curv = float(curvature_features.get("p90_curvature", 0))

    # Normalize: typical max curvature range is [0, 0.5]
    # Values above 0.2 indicate very sharp corners
    sharpness = min(1.0, (max_curv * 3.0 + p90_curv * 2.0) / 2.0)
    return max(0.0, sharpness)


def _compute_boundary_regularity(curvature_features: Dict[str, Any]) -> float:
    """
    Compute boundary irregularity from curvature sign changes.

    Old potholes have highly irregular boundaries (many sign changes in
    curvature) because repeated freeze-thaw cycles and traffic erode
    the edges unevenly.  Fresh potholes have fewer, cleaner breaks.

    Returns a value in [0, 1] where 1 = highly irregular (old).
    """
    sign_changes = float(curvature_features.get("curvature_sign_changes", 0))
    contour_len = float(curvature_features.get("contour_length", 1))

    # Normalize by contour length (longer contours naturally have more changes)
    changes_per_pixel = sign_changes / max(contour_len, 1.0)

    # Typical range: 0.05 (smooth) to 0.3 (very irregular)
    irregularity = min(1.0, changes_per_pixel * 5.0)
    return max(0.0, irregularity)


def _compute_crack_texture_score(
    mask: np.ndarray,
    image_rgb: np.ndarray,
) -> float:
    """
    Compute surrounding crack texture score using Gabor filters.

    Old potholes develop secondary cracks radiating outward from the
    main hole (alligator cracking pattern).  Gabor filters tuned to
    detect linear crack-like features at multiple orientations quantify
    this deterioration pattern.

    Returns a value in [0, 1] where 1 = extensive surrounding cracking (old).
    """
    h, w = mask.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Create surrounding ring (10-30px dilation minus the mask)
    kernel = np.ones((7, 7), np.uint8)
    dilated_small = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    dilated_large = cv2.dilate(mask.astype(np.uint8), kernel, iterations=5)
    ring = dilated_large - dilated_small

    ring_pixels = gray[ring > 0]
    if len(ring_pixels) < 50:
        return 0.0

    # Apply Gabor filters at 4 orientations (0°, 45°, 90°, 135°)
    # Cracks are linear features — Gabor filters detect them optimally
    gabor_responses = []
    for theta_deg in [0, 45, 90, 135]:
        theta = np.radians(theta_deg)
        kernel_gabor = cv2.getGaborKernel(
            ksize=(15, 15),
            sigma=3.0,        # controls Gaussian envelope width
            theta=theta,
            lambd=8.0,        # wavelength — tuned for crack width
            gamma=0.5,        # spatial aspect ratio
            psi=0,
        )
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel_gabor)
        # Mean response in the ring region
        ring_response = np.abs(filtered[ring > 0])
        gabor_responses.append(float(np.mean(ring_response)))

    # Combined crack score: max across orientations (cracks are directional)
    max_response = max(gabor_responses) if gabor_responses else 0.0

    # Normalize: typical Gabor response on clean road is 5-15;
    # cracked road is 25-60
    crack_score = min(1.0, max(0.0, (max_response - 10.0) / 40.0))
    return crack_score


def estimate_pothole_age(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    curvature_features: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Estimate the age of a pothole from visual and geometric cues.

    Physics rationale:
        - Fresh potholes: sharp edges (high curvature), clean breaks
        - Old potholes: rounded edges, irregular boundaries, surrounding
          crack networks (alligator cracking)

    Args:
        mask: Binary pothole mask (HxW).
        image_rgb: Original RGB image (HxW×3).
        curvature_features: Pre-computed curvature features (optional).
                           If None, uses simple contour analysis.

    Returns:
        Dictionary with age_category, age_score, and component scores.
        Returns None on failure.
    """
    try:
        if mask is None or np.sum(mask) < 50:
            return None

        # ── Compute curvature features if not provided ──
        if curvature_features is None:
            try:
                from features import extract_curvature_features
                curvature_features = extract_curvature_features(mask)
            except ImportError:
                curvature_features = {}

        if curvature_features is None:
            curvature_features = {}

        # ── Component scores ──
        edge_sharpness = _compute_edge_sharpness(curvature_features)
        boundary_irregularity = _compute_boundary_regularity(curvature_features)
        crack_texture = _compute_crack_texture_score(mask, image_rgb)

        # Depth-area ratio: deeper potholes relative to their size are usually
        # more established (they've had time to erode downward)
        area = float(np.sum(mask))
        total_area = float(mask.shape[0] * mask.shape[1])
        area_ratio = min(1.0, area / max(total_area * 0.1, 1))

        # ── Weighted age score ──
        # High sharpness = FRESH (invert for age score)
        # High irregularity = OLD
        # High crack texture = OLD
        # Low depth/area ratio = FRESH
        age_score = (
            0.40 * (1.0 - edge_sharpness) +      # smooth edges → old
            0.30 * boundary_irregularity +          # irregular → old
            0.20 * crack_texture +                  # cracks → old
            0.10 * area_ratio                       # large relative area → old
        )
        age_score = max(0.0, min(1.0, age_score))

        # ── Map to category ──
        if age_score < 0.25:
            age_category = "Fresh"
        elif age_score < 0.50:
            age_category = "Developing"
        elif age_score < 0.75:
            age_category = "Established"
        else:
            age_category = "Chronic"

        return {
            "age_category": age_category,
            "age_range": AGE_CATEGORIES[age_category]["range"],
            "age_description": AGE_CATEGORIES[age_category]["description"],
            "age_score": round(age_score, 4),
            "edge_sharpness": round(edge_sharpness, 4),
            "boundary_irregularity": round(boundary_irregularity, 4),
            "crack_texture_score": round(crack_texture, 4),
            "area_ratio": round(area_ratio, 4),
        }

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Severity Progression Prediction
# ═══════════════════════════════════════════════════════════════════════════

# Base deterioration rates (score units per day)
# Grounded in pavement engineering: deeper potholes deteriorate faster
# because exposed base material is weaker than surface asphalt
DETERIORATION_RATES = {
    "Shallow":  0.010,  # slow — surface layer intact
    "Moderate": 0.030,  # medium — base exposed
    "Deep":     0.060,  # fast — subgrade exposed, rapid erosion
}

# Weather multipliers (climate impact on deterioration)
WEATHER_MULTIPLIERS = {
    "freeze_thaw": 1.5,     # water freezes in cracks, expands, accelerates damage
    "tropical_rain": 1.3,   # constant moisture weakens base material
    "temperate": 1.0,       # baseline
    "dry_arid": 0.8,        # minimal water damage
}

# Severity score mapping
SEVERITY_THRESHOLDS = {
    "Shallow":  (0.0, 0.33),
    "Moderate": (0.33, 0.66),
    "Deep":     (0.66, 1.0),
}


def _severity_to_score(severity: str) -> float:
    """Convert severity label to numeric score (midpoint of range)."""
    ranges = SEVERITY_THRESHOLDS.get(severity, (0.33, 0.66))
    return (ranges[0] + ranges[1]) / 2.0


def _score_to_severity(score: float) -> str:
    """Convert numeric score to severity label."""
    score = max(0.0, min(1.0, score))
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= score < hi:
            return label
    return "Deep"


def predict_severity_progression(
    current_severity: str,
    age_estimate: Optional[Dict[str, Any]] = None,
    weather_context: str = "temperate",
) -> Dict[str, Any]:
    """
    Predict how a pothole's severity will change over 30/60/90 days.

    Uses a physics-informed deterioration model based on:
        1. Current severity (deeper = faster degradation)
        2. Estimated age (older = accelerated deterioration)
        3. Climate context (freeze-thaw is worst)

    The model captures the non-linear nature of pavement failure:
    once a pothole reaches "Deep" severity, further deterioration
    is rapid because the protective surface layer is gone and the
    weaker base/subgrade is directly exposed to traffic and weather.

    Args:
        current_severity: "Shallow", "Moderate", or "Deep"
        age_estimate: Output from estimate_pothole_age (optional).
        weather_context: Climate type — see WEATHER_MULTIPLIERS keys.

    Returns:
        Dictionary with current state and 30/60/90 day predictions.
    """
    current_score = _severity_to_score(current_severity)
    base_rate = DETERIORATION_RATES.get(current_severity, 0.03)
    weather_factor = WEATHER_MULTIPLIERS.get(weather_context, 1.0)

    # Age acceleration: older potholes deteriorate faster
    # (more structural damage, more water infiltration paths)
    age_factor = 1.0
    if age_estimate is not None:
        age_score = age_estimate.get("age_score", 0.5)
        # Scale: Fresh (0.25) → 0.8x, Chronic (0.75) → 1.5x
        age_factor = 0.5 + age_score * 1.33
    age_factor = max(0.5, min(2.0, age_factor))

    # Effective deterioration rate
    effective_rate = base_rate * weather_factor * age_factor

    predictions = {}
    for days in [30, 60, 90]:
        # Non-linear progression: rate increases as severity increases
        projected_score = current_score
        for _ in range(days):
            # Daily update: rate scales with current severity
            daily_rate = effective_rate * (0.5 + projected_score)
            projected_score = min(1.0, projected_score + daily_rate)

        predicted_severity = _score_to_severity(projected_score)

        # Confidence decreases with prediction horizon
        confidence = max(0.2, 1.0 - (days / 120.0))

        predictions[f"prediction_{days}d"] = {
            "severity": predicted_severity,
            "score": round(projected_score, 4),
            "confidence": round(confidence, 2),
        }

    return {
        "current_severity": current_severity,
        "current_score": round(current_score, 4),
        "weather_context": weather_context,
        "weather_factor": weather_factor,
        "age_factor": round(age_factor, 4),
        "deterioration_rate_per_day": round(effective_rate, 6),
        **predictions,
    }
