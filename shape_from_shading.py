"""
Shape-from-Shading (SfS) Module
================================
Implements the Frankot & Chellappa (1988) FFT-based gradient integration
algorithm for reconstructing local depth/height fields from shading cues
within pothole masks.

This provides a complementary depth estimate to Depth-Anything-V2:
    - Depth-Anything: global monocular depth (good relative ordering)
    - SfS: local high-frequency shape (concavity, bowl curvature)

The SfS reconstruction captures fine-grained bowl shape that monocular
depth models tend to smooth over, making it useful for severity estimation.

Usage:
    from shape_from_shading import reconstruct_depth_sfs
    height = reconstruct_depth_sfs(gray_image, pothole_mask)
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Illumination Estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_illumination(
    image_gray: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate the dominant illumination direction from gradient statistics
    of the region OUTSIDE the pothole mask (the road surface, assumed
    to be approximately planar and Lambertian).

    Uses the average gradient direction as a proxy for the illumination
    tilt direction (under Lambertian + directional light assumptions).

    Args:
        image_gray: Grayscale image (HxW, uint8 or float).
        mask: Binary pothole mask (HxW).

    Returns:
        (lx, ly): Unit vector of estimated light direction projected
                  onto the image plane.
    """
    gray = image_gray.astype(np.float64)

    # Compute image gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Use only road region (outside mask) for illumination estimation
    road_mask = (mask == 0)

    # Weight by gradient magnitude — strong gradients are more informative
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_road = mag * road_mask

    # Compute weighted average gradient direction
    total_weight = np.sum(mag_road) + 1e-8
    avg_gx = np.sum(gx * mag_road) / total_weight
    avg_gy = np.sum(gy * mag_road) / total_weight

    # Normalize to unit vector
    norm = np.sqrt(avg_gx ** 2 + avg_gy ** 2) + 1e-8
    lx = avg_gx / norm
    ly = avg_gy / norm

    return (float(lx), float(ly))


# ═══════════════════════════════════════════════════════════════════════════
#  Frankot-Chellappa Integration
# ═══════════════════════════════════════════════════════════════════════════

def frankot_chellappa(dzdx: np.ndarray, dzdy: np.ndarray) -> np.ndarray:
    """
    Integrate surface gradient fields (p, q) into a height field Z using
    the Frankot & Chellappa (1988) algorithm.

    The algorithm works in the Fourier domain:
        Z(wx, wy) = (-j*wx*P(wx,wy) - j*wy*Q(wx,wy)) / (wx² + wy²)

    where P = FFT(p), Q = FFT(q), and (wx, wy) are spatial frequencies.

    This is the optimal L2 integrable solution and handles inconsistent
    gradients gracefully (common in real images).

    Args:
        dzdx: Surface gradient in x direction (HxW, float64).
        dzdy: Surface gradient in y direction (HxW, float64).

    Returns:
        height: Integrated height field (HxW, float64), zero-mean.
    """
    h, w = dzdx.shape

    # Spatial frequency grids
    wx = 2 * np.pi * np.fft.fftfreq(w)  # (w,)
    wy = 2 * np.pi * np.fft.fftfreq(h)  # (h,)
    WX, WY = np.meshgrid(wx, wy)        # (h, w)

    # FFT of gradient fields
    P = np.fft.fft2(dzdx)
    Q = np.fft.fft2(dzdy)

    # Denominator (avoid division by zero at DC component)
    denom = WX ** 2 + WY ** 2
    denom[0, 0] = 1.0  # avoid div-by-zero; DC will be set to 0

    # Frankot-Chellappa formula
    Z_fft = (-1j * WX * P - 1j * WY * Q) / denom
    Z_fft[0, 0] = 0.0  # zero-mean constraint

    # Inverse FFT → real height field
    height = np.real(np.fft.ifft2(Z_fft))

    return height


# ═══════════════════════════════════════════════════════════════════════════
#  Full SfS Reconstruction Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_depth_sfs(
    image_gray: np.ndarray,
    mask: np.ndarray,
    illumination_direction: Optional[Tuple[float, float]] = None,
    albedo: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Reconstruct the local depth/height field inside a pothole mask using
    Shape-from-Shading with Frankot-Chellappa integration.

    Assumes:
        - Lambertian reflectance model: I = albedo * (n · l)
        - Directional illumination (estimated from surrounding road)
        - The pothole is a concave surface in an otherwise flat road

    The output is a relative height field (not metric depth), useful for:
        - Bowl shape characterization
        - Curvature feature extraction
        - Severity estimation independent of monocular depth model

    Args:
        image_gray: Grayscale image (HxW, uint8 or float).
        mask: Binary pothole mask (HxW, {0, 1}).
        illumination_direction: (lx, ly) unit vector. If None, estimated
                                automatically from surrounding road.
        albedo: Assumed surface albedo (default 0.5). In practice this
                is a scale factor; relative shape is preserved.

    Returns:
        height: Height field (HxW, float64) where pothole interior has
                reconstructed depth values and exterior is zero.
                Returns None on failure.
    """
    try:
        if mask is None or np.sum(mask) < 50:
            return None

        gray = image_gray.astype(np.float64) / 255.0
        h, w = gray.shape

        # ── Estimate illumination if not provided ──
        if illumination_direction is None:
            lx, ly = estimate_illumination(image_gray, mask)
        else:
            lx, ly = illumination_direction

        # Light direction vector (lx, ly, lz)
        # Assume light comes from above at ~45° elevation
        lz = np.sqrt(max(0, 1.0 - lx ** 2 - ly ** 2))
        if lz < 0.1:
            lz = 0.5
            norm = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
            lx, ly, lz = lx / norm, ly / norm, lz / norm

        # ── Compute intensity gradients ──
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # ── Estimate surface gradients (p, q) from Lambertian model ──
        # Under Lambertian: I = albedo * (n · l)
        # With n ≈ (-p, -q, 1) / sqrt(p²+q²+1):
        #   I ≈ albedo * (-p*lx - q*ly + lz) / sqrt(p²+q²+1)
        # For small slopes (|p|, |q| << 1):
        #   dI/dx ≈ albedo * (-dp/dx * lx)  →  p ≈ -gx / (albedo * lx)
        #   dI/dy ≈ albedo * (-dq/dy * ly)  →  q ≈ -gy / (albedo * ly)

        eps = 1e-6
        if abs(lx) > eps:
            p = -gx / (albedo * lx + eps)
        else:
            p = -gx / (albedo * 0.5)

        if abs(ly) > eps:
            q = -gy / (albedo * ly + eps)
        else:
            q = -gy / (albedo * 0.5)

        # ── Mask the gradients to pothole region ──
        # Expand mask slightly for smooth boundary handling
        kernel = np.ones((5, 5), np.uint8)
        mask_expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

        p_masked = p * mask_expanded.astype(np.float64)
        q_masked = q * mask_expanded.astype(np.float64)

        # ── Frankot-Chellappa integration ──
        height = frankot_chellappa(p_masked, q_masked)

        # ── Post-processing ──
        # Zero out exterior
        height = height * mask.astype(np.float64)

        # Normalize to reasonable range inside mask
        interior = height[mask > 0]
        if len(interior) > 0 and np.std(interior) > 1e-8:
            # Normalize to [0, 1] inside mask
            h_min = np.percentile(interior, 2)
            h_max = np.percentile(interior, 98)
            if h_max > h_min:
                height = (height - h_min) / (h_max - h_min)
                height = np.clip(height, 0, 1) * mask.astype(np.float64)

        return height

    except Exception:
        return None
