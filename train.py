# converts rtMRI data into AI training data, trains the hybrid CNN, and evaluates voiced/unvoiced classification performance.

# MAT + AVI
#     ↓
# Preprocessing
#     ↓
# Physiological Features
#     ↓
# ROI Images
#     ↓
# Dataset Creation
#     ↓
# CNN Training
#     ↓
# Evaluation

import os
import math
import cv2
import numpy as np

from scipy.io import loadmat
from scipy.interpolate import interp1d

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch

from preprocessing.loader import (
    load_annotations,
    load_video,
    extract,
    densify_contour,
    filter_outliers
)

from preprocessing.endpoint import detect_endpoint

from preprocessing.circle import (
    find_bulge_contour_segment,
    two_point_anchored_circle,
    extract_circle_roi,
    distance
)

from config import *


# ==========================================================
# INTERPOLATE CONTOUR
# ==========================================================
def contour_interpolator(contour):

    contour = contour[np.argsort(contour[:, 1])]

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
# COMPUTE AIR/TISSUE REGIONS
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

    contour2_interp = contour_interpolator(contour2)

    contour3_interp = contour_interpolator(contour3)

    air_pixels = 0

    tissue_pixels = 0

    for y in range(h):

        for x in range(w):

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

            if abs(x - x_c2) < 1:
                continue 

            if abs(x - x_c3) < 1:
                continue

            # REGION C = TISSUE
            if x > x_c3:

                tissue_pixels += 1

            # REGION B = AIR
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

    air_tissue_ratio = (
        air_pixels /
        (tissue_pixels + epsilon)
    )

    glottis_length = distance(
        GLTB,
        endpoint
    )

    circle_area = math.pi * (radius ** 2)

    curvature = np.std(
        np.diff(contour3[:, 0])
    )

    line_vec = endpoint - GLTB

    norm = np.linalg.norm(line_vec)

    if norm < 1e-6:

        bulge_depth = 0

    else:

        depths = []

        for p in contour3:

            depth = np.abs(
                np.cross(line_vec, p - GLTB)
            ) / norm

            depths.append(depth)

        bulge_depth = np.max(depths)

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

    # voiced
    if tissue_pixels > air_pixels:
        return 1

    # unvoiced
    return 0


# ==========================================================
# BUILD DATASET
# ==========================================================
def build_dataset(data_folder):

    image_data = []

    feature_data = []

    labels = []

    mat_files = [
        f for f in os.listdir(data_folder)
        if f.endswith('.mat')
    ]

    for mat_name in mat_files:

        base = os.path.splitext(mat_name)[0]

        avi_name = base + ".avi"

        mat_path = os.path.join(
            data_folder,
            mat_name
        )

        avi_path = os.path.join(
            data_folder,
            avi_name
        )

        if not os.path.exists(avi_path):
            continue

        print(f"\nProcessing {base}")

        print("Current MAT file:", mat_path)
        
        annotations = load_annotations(mat_path)

        cap = load_video(avi_path)

        frame_idx = 0

        while True:

            ret, frame = cap.read()

            if (
                not ret or
                frame_idx >= len(annotations)
            ):
                break

            ann = annotations[frame_idx]

            contour2 = extract(ann['contour2'])

            contour3 = extract(ann['contour3'])

            GLTB = extract(ann['GLTB'])

            if (
                contour2.size == 0 or
                contour3.size == 0
            ):
                frame_idx += 1
                continue

            contour2 = contour2 - 1

            contour3 = contour3 - 1

            GLTB = GLTB.flatten() - 1

            contour3_original = contour3.copy()

            contour3_processed = densify_contour(
                contour3
            )

            contour3_processed = filter_outliers(
                contour3_processed
            )

            endpoint = detect_endpoint(
                contour3_processed,
                contour3_original,
                GLTB
            )

            if endpoint is None:

                frame_idx += 1

                continue

            bulge_points = (
                find_bulge_contour_segment(
                    contour3_original,
                    GLTB,
                    endpoint
                )
            )

            circle_result = (
                two_point_anchored_circle(
                    GLTB,
                    endpoint,
                    bulge_points
                )
            )

            if circle_result is None:

                frame_idx += 1

                continue

            center, radius = circle_result

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            air_pixels, tissue_pixels = (
                compute_air_tissue_regions(
                    frame_gray,
                    contour2,
                    contour3_original,
                    center,
                    radius
                )
            )

            features = (
                compute_physiological_features(
                    air_pixels,
                    tissue_pixels,
                    GLTB,
                    endpoint,
                    radius,
                    contour3_original
                )
            )

            label = generate_label(
                air_pixels,
                tissue_pixels
            )

            roi = extract_circle_roi(
                frame_rgb,
                center,
                radius,
                ROI_SIZE
            )

            if roi is None:

                frame_idx += 1

                continue

            image_data.append(roi)

            feature_data.append(features)

            labels.append(label)

            frame_idx += 1

        cap.release()

    image_data = np.array(image_data)

    feature_data = np.array(feature_data)

    labels = np.array(labels)

    print("\nDataset Built")

    print("Images :", image_data.shape)

    print("Features:", feature_data.shape)

    print("Labels :", labels.shape)

    return (
        image_data,
        feature_data,
        labels
    )


# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_model(
    model,
    train_loader,
    optimizer,
    criterion
):

    model.train()

    running_loss = 0

    for images, features, labels in train_loader:

        images = images.to(DEVICE)

        features = features.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            images,
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# ==========================================================
# EVALUATION
# ==========================================================
def evaluate_model(
    model,
    test_loader
):

    model.eval()

    preds = []

    trues = []

    with torch.no_grad():

        for images, features, labels in test_loader:

            images = images.to(DEVICE)

            features = features.to(DEVICE)

            outputs = model(
                images,
                features
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            preds.extend(
                predicted.cpu().numpy()
            )

            trues.extend(
                labels.numpy()
            )

    acc = accuracy_score(
        trues,
        preds
    )

    prec = precision_score(
        trues,
        preds
    )

    rec = recall_score(
        trues,
        preds
    )

    f1 = f1_score(
        trues,
        preds
    )

    print("\n====================")

    print("RESULTS")

    print("====================")

    print("Accuracy :", acc)

    print("Precision:", prec)

    print("Recall   :", rec)

    print("F1 Score :", f1)

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            trues,
            preds
        )
    )

    print("\nClassification Report")

    print(
        classification_report(
            trues,
            preds
        )
    )