# converts glottis geometry into physiological features and voiced/unvoiced labels for AI training.
    
#     Contours + Circle
#         ↓
# Air/Tissue Regions
#         ↓
# Physiological Measurements
#         ↓
# Feature Vector
#         ↓
# Voiced/Unvoiced Label

import math
import numpy as np

from scipy.interpolate import interp1d

from circle import distance


# ==========================================================
# INTERPOLATE CONTOUR X(y)
# ==========================================================
def contour_interpolator(contour):

    contour = contour[
        np.argsort(contour[:, 1])
    ]

    ys = contour[:, 1]

    xs = contour[:, 0]

    unique_y, idx = np.unique(
        ys,
        return_index=True
    )

    unique_x = xs[idx]

    return interp1d(
        unique_y,
        unique_x,
        bounds_error=False,
        fill_value="extrapolate"
    )


# ==========================================================
# COMPUTE REGION A/B/C PIXELS
# ==========================================================
def compute_air_tissue_regions(
    frame_gray,
    contour2,
    contour3,
    center,
    radius
):

    h, w = frame_gray.shape

    cx, cy = center

    contour2_interp = contour_interpolator(
        contour2
    )

    contour3_interp = contour_interpolator(
        contour3
    )

    air_pixels = 0

    tissue_pixels = 0

    for y in range(h):

        for x in range(w):

            # ------------------------------------------
            # INSIDE CIRCLE ONLY
            # ------------------------------------------
            d = math.sqrt(
                (x - cx) ** 2 +
                (y - cy) ** 2
            )

            if d >= radius:
                continue

            try:

                x_c2 = contour2_interp(y)

                x_c3 = contour3_interp(y)

            except:
                continue

            # ------------------------------------------
            # IGNORE BOUNDARIES
            # ------------------------------------------
            if abs(x - x_c2) < 1:
                continue

            if abs(x - x_c3) < 1:
                continue

            # ------------------------------------------
            # REGION C = TISSUE
            # ------------------------------------------
            if x > x_c3:

                tissue_pixels += 1

            # ------------------------------------------
            # REGION B = AIR
            # ------------------------------------------
            elif x_c2 <= x < x_c3:

                air_pixels += 1

    return air_pixels, tissue_pixels


# ==========================================================
# PHYSIOLOGICAL FEATURES
# ==========================================================
def compute_physiological_features(
    air_pixels,
    tissue_pixels,
    GLTB,
    endpoint,
    radius,
    contour3
):

    epsilon = 1e-6

    # ----------------------------------------------
    # AIR/TISSUE RATIO
    # ----------------------------------------------
    air_tissue_ratio = (
        air_pixels /
        (tissue_pixels + epsilon)
    )

    # ----------------------------------------------
    # GLOTTIS LENGTH
    # ----------------------------------------------
    glottis_length = distance(
        GLTB,
        endpoint
    )

    # ----------------------------------------------
    # CIRCLE AREA
    # ----------------------------------------------
    circle_area = math.pi * (
        radius ** 2
    )

    # ----------------------------------------------
    # CURVATURE APPROXIMATION
    # ----------------------------------------------
    curvature = np.std(
        np.diff(contour3[:, 0])
    )

    # ----------------------------------------------
    # BULGE DEPTH
    # ----------------------------------------------
    line_vec = endpoint - GLTB

    norm = np.linalg.norm(line_vec)

    if norm < 1e-6:

        bulge_depth = 0

    else:

        depths = []

        for p in contour3:

            depth = np.abs(
                np.cross(
                    line_vec,
                    p - GLTB
                )
            ) / norm

            depths.append(depth)

        bulge_depth = np.max(depths)

    # ----------------------------------------------
    # FEATURE VECTOR
    # ----------------------------------------------
    features = np.array([

        air_pixels,

        tissue_pixels,

        air_tissue_ratio,

        glottis_length,

        radius,

        circle_area,

        curvature,

        bulge_depth

    ], dtype=np.float32)

    return features


# ==========================================================
# LABEL GENERATION
# ==========================================================
def generate_label(
    air_pixels,
    tissue_pixels
):

    # ---------------------------------------------
    # VOICED = tissue > air
    # UNVOICED = air > tissue
    # ---------------------------------------------

    if tissue_pixels > air_pixels:
        return 1

    return 0