"""
Adverse Condition Synthesizer
=============================
Generates synthetic adverse weather and lighting variants of road images
for robustness benchmarking.  Every function works on a single image,
returns a modified **copy** (never mutates input), and uses only
numpy / OpenCV — no extra dependencies.

Supported conditions:
    rain, night, fog, waterlogged, shadow

Usage:
    from adverse_conditions import synthesize_condition, get_available_conditions
    variant = synthesize_condition(image_bgr, "rain", seed=42)
"""

from typing import List, Optional

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def get_available_conditions() -> List[str]:
    """Return the list of supported adverse condition names."""
    return ["rain", "night", "fog", "waterlogged", "shadow"]


def synthesize_condition(
    image: np.ndarray,
    condition: str,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply a synthetic adverse condition to an image.

    Args:
        image:     BGR uint8 image (HxWx3).
        condition: One of get_available_conditions().
        mask:      Binary pothole mask (HxW, uint8 {0,1}).
                   Required for "waterlogged"; ignored otherwise.
        seed:      Optional RNG seed for reproducibility.

    Returns:
        Modified copy of the image (BGR uint8).

    Raises:
        ValueError: Unknown condition or missing mask for waterlogged.
    """
    if seed is not None:
        np.random.seed(seed)

    condition = condition.lower().strip()
    out = image.copy()

    if condition == "rain":
        return _apply_rain(out)
    elif condition == "night":
        return _apply_night(out)
    elif condition == "fog":
        return _apply_fog(out)
    elif condition == "waterlogged":
        if mask is None:
            raise ValueError("waterlogged condition requires a mask parameter")
        return _apply_waterlogged(out, mask)
    elif condition == "shadow":
        return _apply_shadow(out)
    else:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Available: {get_available_conditions()}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Internal Synthesizers
# ═══════════════════════════════════════════════════════════════════════════

def _apply_rain(image: np.ndarray) -> np.ndarray:
    """
    Simulate rain on the image.

    1. Generate directional rain streaks (near-vertical, angle 70-110°)
    2. Add light Gaussian sensor noise (simulates camera noise in rain)
    3. Increase overall brightness by 5-15% (wet surfaces reflect more)
    4. Blend rain streak overlay at 20-30% opacity
    """
    h, w = image.shape[:2]

    # ── Rain streaks ──
    streak_layer = np.zeros((h, w), dtype=np.uint8)
    num_streaks = int(np.random.uniform(300, 700) * (h * w) / (640 * 640))
    angle_deg = np.random.uniform(75, 105)
    angle_rad = np.radians(angle_deg)

    for _ in range(num_streaks):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(15, 45)
        dx = int(length * np.cos(angle_rad))
        dy = int(length * np.sin(angle_rad))
        brightness = np.random.randint(160, 230)
        thickness = 1
        cv2.line(streak_layer, (x, y), (x + dx, y + dy), int(brightness), thickness)

    # Blur streaks slightly for realism
    streak_layer = cv2.GaussianBlur(streak_layer, (3, 1), sigmaX=0.5)

    # ── Blend rain ──
    opacity = np.random.uniform(0.20, 0.30)
    streak_bgr = cv2.cvtColor(streak_layer, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(image, 1.0, streak_bgr, opacity, 0)

    # ── Sensor noise ──
    noise = np.random.normal(0, 8, result.shape).astype(np.float32)
    result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # ── Brightness boost (wet road reflection) ──
    boost = np.random.uniform(1.05, 1.15)
    result = np.clip(result.astype(np.float32) * boost, 0, 255).astype(np.uint8)

    return result


def _apply_night(image: np.ndarray) -> np.ndarray:
    """
    Simulate nighttime driving conditions.

    1. Reduce overall brightness to 15-25% of original
    2. Add Poisson-like sensor noise (low-light camera noise)
    3. Optional headlight cone (bright elliptical region at center-bottom)
    """
    h, w = image.shape[:2]

    # ── Darken ──
    darkness_factor = np.random.uniform(0.15, 0.25)
    result = (image.astype(np.float32) * darkness_factor)

    # ── Headlight cone ──
    # Simulate front-facing headlights illuminating road ahead
    headlight = np.zeros((h, w), dtype=np.float32)
    center_x = w // 2
    center_y = int(h * 0.85)  # near bottom of image
    axes_x = int(w * np.random.uniform(0.3, 0.5))
    axes_y = int(h * np.random.uniform(0.3, 0.45))
    cv2.ellipse(
        headlight,
        (center_x, center_y),
        (axes_x, axes_y),
        0, 0, 360,
        1.0, -1,
    )
    headlight = cv2.GaussianBlur(headlight, (0, 0), sigmaX=axes_x * 0.4)
    headlight = headlight / (headlight.max() + 1e-8)

    # Headlight adds brightness back (up to 60% in center)
    illumination = np.random.uniform(0.4, 0.7)
    for c in range(3):
        result[:, :, c] += image[:, :, c].astype(np.float32) * headlight * illumination

    result = np.clip(result, 0, 255)

    # ── Sensor noise (Poisson-like) ──
    # In low light, noise is more visible
    noise_scale = np.random.uniform(8, 15)
    noise = np.random.normal(0, noise_scale, result.shape)
    result = np.clip(result + noise, 0, 255).astype(np.uint8)

    # ── Slight blue-shift (sodium vapor street lighting effect) ──
    result_float = result.astype(np.float32)
    result_float[:, :, 0] = np.clip(result_float[:, :, 0] * 1.1, 0, 255)  # blue
    result_float[:, :, 2] = np.clip(result_float[:, :, 2] * 0.9, 0, 255)  # red
    result = result_float.astype(np.uint8)

    return result


def _apply_fog(image: np.ndarray) -> np.ndarray:
    """
    Simulate foggy conditions.

    1. Create uniform gray fog layer (value 180-220)
    2. Add slight depth-dependent fog density (objects farther = more fog)
    3. Blend with original at 40-60% opacity
    4. Slight Gaussian blur to reduce contrast
    """
    h, w = image.shape[:2]

    # ── Fog layer (uniform gray with slight variation) ──
    fog_value = np.random.randint(180, 220)
    fog_layer = np.full_like(image, fog_value, dtype=np.uint8)

    # ── Depth-dependent density (top of image = farther = more fog) ──
    depth_gradient = np.linspace(0.7, 0.3, h).reshape(-1, 1)  # top=dense, bottom=less
    depth_gradient = np.tile(depth_gradient, (1, w))

    # ── Blend ──
    base_opacity = np.random.uniform(0.40, 0.60)
    opacity_map = (base_opacity * depth_gradient).astype(np.float32)
    opacity_map = np.stack([opacity_map] * 3, axis=-1)

    result = (image.astype(np.float32) * (1 - opacity_map) +
              fog_layer.astype(np.float32) * opacity_map)

    # ── Slight blur (fog reduces sharpness) ──
    sigma = np.random.uniform(1.0, 2.0)
    result = cv2.GaussianBlur(result.astype(np.uint8), (0, 0), sigmaX=sigma)

    # ── Reduce contrast ──
    mean_val = np.mean(result)
    contrast_reduction = np.random.uniform(0.15, 0.25)
    result = result.astype(np.float32)
    result = result * (1 - contrast_reduction) + mean_val * contrast_reduction

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_waterlogged(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Simulate water filling the pothole.

    1. Sample sky/background color from top 10% of image
    2. Replace mask interior with sky-tinted reflection
    3. Add 3-5 specular highlight spots (small bright ellipses)
    4. Slight blur inside mask for water surface effect
    5. Smooth transition at mask boundary

    This directly tests the hypothesis: geometry features should be
    robust to waterlogging while depth features degrade.
    """
    h, w = image.shape[:2]
    result = image.copy()

    if mask is None or np.sum(mask) == 0:
        return result

    # ── Sample sky color from top 10% of image ──
    top_rows = max(1, h // 10)
    sky_region = image[:top_rows, :, :]
    sky_color = np.mean(sky_region, axis=(0, 1)).astype(np.float32)

    # Desaturate slightly (water doesn't perfectly reflect)
    sky_gray = np.mean(sky_color)
    sky_color = sky_color * 0.6 + sky_gray * 0.4

    # ── Create water surface ──
    water_surface = np.full_like(image, sky_color.astype(np.uint8), dtype=np.uint8)

    # Add slight color variation (water ripple effect)
    ripple = np.random.normal(0, 10, (h, w)).astype(np.float32)
    ripple = cv2.GaussianBlur(ripple, (0, 0), sigmaX=5)
    for c in range(3):
        water_surface[:, :, c] = np.clip(
            water_surface[:, :, c].astype(np.float32) + ripple, 0, 255
        ).astype(np.uint8)

    # ── Blend water surface into mask region ──
    mask_float = mask.astype(np.float32)

    # Erode mask slightly for smooth transition at edges
    kernel_small = np.ones((3, 3), np.uint8)
    mask_eroded = cv2.erode(mask.astype(np.uint8), kernel_small, iterations=2)
    mask_smooth = cv2.GaussianBlur(mask_eroded.astype(np.float32), (11, 11), sigmaX=3)
    mask_smooth = np.clip(mask_smooth, 0, 1)

    # Blend: 70% water, 30% original road (road texture slightly visible through water)
    water_opacity = np.random.uniform(0.65, 0.80)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c].astype(np.float32) * (1 - mask_smooth * water_opacity) +
            water_surface[:, :, c].astype(np.float32) * mask_smooth * water_opacity
        ).astype(np.uint8)

    # ── Specular highlights (3-5 bright spots) ──
    num_highlights = np.random.randint(3, 6)
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        for _ in range(num_highlights):
            idx = np.random.randint(0, len(xs))
            cx, cy = int(xs[idx]), int(ys[idx])
            radius_x = np.random.randint(3, max(4, int(np.sqrt(np.sum(mask)) * 0.05)))
            radius_y = np.random.randint(2, max(3, radius_x))
            angle = np.random.uniform(0, 360)
            cv2.ellipse(
                result,
                (cx, cy),
                (radius_x, radius_y),
                angle, 0, 360,
                (240, 245, 250),  # near-white specular
                -1,
            )

    # ── Slight blur inside mask (water surface smoothness) ──
    blurred = cv2.GaussianBlur(result, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_smooth] * 3, axis=-1)
    result = (result.astype(np.float32) * (1 - mask_3ch * 0.5) +
              blurred.astype(np.float32) * mask_3ch * 0.5)

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_shadow(image: np.ndarray) -> np.ndarray:
    """
    Simulate partial shadow across the road.

    1. Create gradient darkening mask (random orientation)
    2. Darken one region to 30-50% brightness
    3. Smooth transition zone (10-20% of image dimension)
    """
    h, w = image.shape[:2]

    # ── Shadow direction (left-right or top-bottom) ──
    direction = np.random.choice(["left", "right", "top", "bottom"])
    shadow_intensity = np.random.uniform(0.30, 0.50)
    transition_frac = np.random.uniform(0.10, 0.20)

    shadow_map = np.ones((h, w), dtype=np.float32)

    if direction in ("left", "right"):
        transition_px = int(w * transition_frac)
        midpoint = np.random.randint(int(w * 0.3), int(w * 0.7))

        for x in range(w):
            if direction == "left":
                if x < midpoint - transition_px // 2:
                    shadow_map[:, x] = shadow_intensity
                elif x < midpoint + transition_px // 2:
                    t = (x - (midpoint - transition_px // 2)) / max(transition_px, 1)
                    shadow_map[:, x] = shadow_intensity + t * (1 - shadow_intensity)
            else:
                if x > midpoint + transition_px // 2:
                    shadow_map[:, x] = shadow_intensity
                elif x > midpoint - transition_px // 2:
                    t = ((midpoint + transition_px // 2) - x) / max(transition_px, 1)
                    shadow_map[:, x] = shadow_intensity + t * (1 - shadow_intensity)
    else:
        transition_px = int(h * transition_frac)
        midpoint = np.random.randint(int(h * 0.3), int(h * 0.7))

        for y in range(h):
            if direction == "top":
                if y < midpoint - transition_px // 2:
                    shadow_map[y, :] = shadow_intensity
                elif y < midpoint + transition_px // 2:
                    t = (y - (midpoint - transition_px // 2)) / max(transition_px, 1)
                    shadow_map[y, :] = shadow_intensity + t * (1 - shadow_intensity)
            else:
                if y > midpoint + transition_px // 2:
                    shadow_map[y, :] = shadow_intensity
                elif y > midpoint - transition_px // 2:
                    t = ((midpoint + transition_px // 2) - y) / max(transition_px, 1)
                    shadow_map[y, :] = shadow_intensity + t * (1 - shadow_intensity)

    # ── Apply shadow ──
    shadow_map_3ch = np.stack([shadow_map] * 3, axis=-1)
    result = (image.astype(np.float32) * shadow_map_3ch)

    return np.clip(result, 0, 255).astype(np.uint8)
