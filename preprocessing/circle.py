

import numpy as np
import cv2


# ==========================================================
# TEMPORAL MEMORY
# ==========================================================
prev_length = None
prev_prev_length = None


# ==========================================================
# DISTANCE
# ==========================================================
def distance(p1, p2):

    return np.linalg.norm(np.array(p1) - np.array(p2))


# ==========================================================
# APPLY TEMPORAL CONSTRAINT
# ==========================================================
def apply_temporal_constraint(
    candidate_points,
    contour_original,
    start_point,
    end_point
):

    global prev_length
    global prev_prev_length

    gltb_x = start_point[0]
    gltb_y = start_point[1]

    current_length = np.sqrt(
        (end_point[0] - gltb_x) ** 2 +
        (end_point[1] - gltb_y) ** 2
    )

    anomaly_detected = False

    if (
        prev_length is not None and
        prev_prev_length is not None
    ):

        if (
            current_length > 1.35 * prev_length and
            current_length > 1.35 * prev_prev_length
        ):
            anomaly_detected = True

    if anomaly_detected:

        target_length = prev_length

        best_point = None

        best_error = float('inf')

        for pt in candidate_points:

            px, py = pt

            dist = np.sqrt(
                (px - gltb_x) ** 2 +
                (py - gltb_y) ** 2
            )

            error = abs(dist - target_length)

            if error < best_error:

                best_error = error

                best_point = pt

        if best_point is not None:

            temp_best = np.array(
                best_point,
                dtype=float
            )

            distances = np.linalg.norm(
                contour_original - temp_best,
                axis=1
            )

            closest_idx = np.argmin(distances)

            end_point = contour_original[closest_idx]

            current_length = target_length

    prev_prev_length = prev_length

    prev_length = current_length

    return end_point


# ==========================================================
# FIND BULGE SEGMENT
# ==========================================================
def find_bulge_contour_segment(contour, start, end):

    P = np.array(contour, dtype=float)

    S = np.array(start, dtype=float)

    E = np.array(end, dtype=float)

    if len(P) < 2:
        return np.empty((0, 2))

    start_idx = np.argmin(np.linalg.norm(P - S, axis=1))

    end_idx = np.argmin(np.linalg.norm(P - E, axis=1))

    i1 = min(start_idx, end_idx)

    i2 = max(start_idx, end_idx)

    segment = P[i1:i2 + 1]

    return segment


# ==========================================================
# TWO POINT ANCHORED CIRCLE
# ==========================================================
T_MIN = -100
T_MAX = 100
T_STEP = 0.10
INSIDE_TOLERANCE = 1.5


def two_point_anchored_circle(
    start,
    end,
    bulge_points,
    candidate_points=None,
    contour_original=None
):

    # ======================================================
    # APPLY TEMPORAL CONSTRAINT
    # ======================================================
    if (
        candidate_points is not None and
        contour_original is not None
    ):

        end = apply_temporal_constraint(
            candidate_points=candidate_points,
            contour_original=contour_original,
            start_point=start,
            end_point=end
        )

    S = np.array(start, dtype=float)

    E = np.array(end, dtype=float)

    B = np.array(bulge_points, dtype=float)

    chord_vec = E - S

    chord_len = np.linalg.norm(chord_vec)

    if chord_len < 1e-6:
        return None

    midpoint = (S + E) / 2.0

    perp = np.array([
        -chord_vec[1],
         chord_vec[0]
    ], dtype=float)

    perp = perp / np.linalg.norm(perp)

    best_center = None

    best_radius = None

    for t in np.arange(T_MIN, T_MAX + T_STEP, T_STEP):

        center = midpoint + t * perp

        radius = distance(center, S)

        if len(B) > 0:

            dists = np.linalg.norm(
                B - center,
                axis=1
            )

            valid = np.all(
                dists <= radius + INSIDE_TOLERANCE
            )

        else:

            valid = True

        if valid:

            if (
                best_radius is None or
                radius < best_radius
            ):

                best_center = center

                best_radius = radius

    if best_center is None:
        return None

    return best_center, best_radius


# ==========================================================
# EXTRACT CIRCLE ROI
# ==========================================================
def extract_circle_roi(
    frame,
    center,
    radius,
    roi_size=224
):

    cx, cy = map(int, center)

    r = int(radius)

    x1 = max(cx - r, 0)

    y1 = max(cy - r, 0)

    x2 = min(cx + r, frame.shape[1])

    y2 = min(cy + r, frame.shape[0])

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    roi = cv2.resize(
        roi,
        (roi_size, roi_size)
    )

    return roi














