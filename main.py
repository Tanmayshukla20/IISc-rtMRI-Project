# controls the complete biomedical AI pipeline from data loading to final voiced/unvoiced prediction model training.

# main.py
#    ↓
# Build Dataset
#    ↓
# Train/Test Split
#    ↓
# Create DataLoaders
#    ↓
# Build Hybrid CNN
#    ↓
# Train AI
#    ↓
# Evaluate AI
#    ↓
# Save Model

from train import (
    build_dataset,
    train_model,
    evaluate_model
)

from model.dataset import HybridDataset
from model.hybrid_model import HybridCNN

from config import *

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    print("Using device:", DEVICE)
    # BUILD DATASET
    images, features, labels = build_dataset(DATA_FOLDER)

    # SPLIT
    X_train_img, X_test_img, \
    X_train_feat, X_test_feat, \
    y_train, y_test = train_test_split(
        images,
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # DATASETS
    train_dataset = HybridDataset(
        X_train_img,
        X_train_feat,
        y_train
    )

    test_dataset = HybridDataset(
        X_test_img,
        X_test_feat,
        y_test
    )

    # DATALOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # MODEL
    model = HybridCNN(
        num_features=X_train_feat.shape[1]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # TRAINING
    for epoch in range(EPOCHS):

        loss = train_model(
            model,
            train_loader,
            optimizer,
            criterion
        )

        print(f"Epoch {epoch+1}: {loss:.4f}")

    # EVALUATION
    evaluate_model(model, test_loader)

    # SAVE MODEL
    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH
    )

    print("Model Saved")


if __name__ == "__main__":

    main()