# controls all important settings and paths of the biomedical AI pipeline.

import torch

DATA_FOLDER = "./dataset"

ROI_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 2

LEARNING_RATE = 1e-4

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MODEL_SAVE_PATH = "./outputs/model.pth"

RESULTS_PATH = "./outputs/results.txt"