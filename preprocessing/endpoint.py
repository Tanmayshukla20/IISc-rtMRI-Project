# detects the glottis endpoint and stabilizes it across frames using temporal memory constraints.

# Contour3
#     ↓
# Scanline Analysis
#     ↓
# Endpoint Detection
#     ↓
# Temporal Anomaly Check
#     ↓
# Corrected Stable Endpoint

import numpy as np
from collections import defaultdict


prev_length = None
prev_prev_length = None


def detect_endpoint(contour3_processed,
                    contour3_original,
                    GLTB):

    global prev_length
    global prev_prev_length

    gltb_x = int(GLTB[0])
    gltb_y = int(GLTB[1])

    contour = contour3_processed[
        contour3_processed[:,1] > gltb_y
    ]

    if len(contour) == 0:
        return None

    contour_by_y = defaultdict(list)

    for (x,y) in contour:
        contour_by_y[int(y)].append(int(x))

    y_values = sorted(contour_by_y.keys())

    y_top = y_values[0]

    x_left = min(contour_by_y[y_top])

    x_right = gltb_x

    min_rows_after_gltb = 6
    min_depth_pixels = 12

    row_counter = 0

    net_values = []
    candidate_points = []

    for yk in y_values:

        if yk <= y_top:
            continue

        row_counter += 1

        if row_counter < min_rows_after_gltb:
            continue

        if (yk - y_top) < min_depth_pixels:
            continue

        net_sum = 0

        for y in range(y_top, yk):

            if y not in contour_by_y:
                continue

            x_boundary = min(contour_by_y[y])

            for x in range(x_left, x_right):

                if x < x_boundary:
                    net_sum -= 1
                else:
                    net_sum += 1

        net_values.append(net_sum)

        x_real = min(contour_by_y[yk])

        candidate_points.append((x_real, yk))

    if len(net_values) == 0:
        return None

    net_values = np.array(net_values)

    idx = np.argmin(np.abs(net_values))

    end_point = np.array(candidate_points[idx], dtype=float)

    distances = np.linalg.norm(
        contour3_original - end_point,
        axis=1
    )

    closest_idx = np.argmin(distances)

    end_point = contour3_original[closest_idx]

    current_length = np.sqrt(
        (end_point[0]-gltb_x)**2 +
        (end_point[1]-gltb_y)**2
    )

    anomaly_detected = False

    if prev_length is not None and prev_prev_length is not None:

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
                (px-gltb_x)**2 +
                (py-gltb_y)**2
            )

            error = abs(dist - target_length)

            if error < best_error:
                best_error = error
                best_point = pt

        if best_point is not None:

            temp_best = np.array(best_point, dtype=float)

            distances = np.linalg.norm(
                contour3_original - temp_best,
                axis=1
            )

            closest_idx = np.argmin(distances)

            end_point = contour3_original[closest_idx]

            current_length = target_length

    prev_prev_length = prev_length
    prev_length = current_length

    return end_point