# loads the rtMRI data and cleans the contour points before analysis.

# MAT file + Video
#         ↓
# Extract contours
#         ↓
# Densify contour
#         ↓
# Remove noisy points
#         ↓
# Clean contour for processing

import numpy as np
import cv2
from scipy.io import loadmat


def load_annotations(mat_path):

    mat = loadmat(mat_path)

    return mat['annotation'][0]


def load_video(video_path):

    cap = cv2.VideoCapture(video_path)

    return cap


def extract(val):


    while isinstance(val, np.ndarray) and val.dtype == 'O':
        val = val[0]

    return np.array(val)


def densify_contour(contour):

    contour = contour.astype(float)

    contour = contour[np.argsort(contour[:,1])]

    new_pts = []

    for i in range(len(contour)-1):

        p1 = contour[i]
        p2 = contour[i+1]

        new_pts.append(p1)
        new_pts.append((p1+p2)/2.0)

    new_pts.append(contour[-1])

    return np.array(new_pts)


def filter_outliers(contour):

    if len(contour) < 5:
        return contour

    contour = contour[np.argsort(contour[:,1])]

    diffs = np.linalg.norm(np.diff(contour, axis=0), axis=1)

    thresh = np.median(diffs) * 3

    mask = np.ones(len(contour), dtype=bool)

    mask[1:] = diffs < thresh

    return contour[mask]