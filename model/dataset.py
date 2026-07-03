#prepares your biomedical data for neural network training.

# Raw Data
#    ↓
# HybridDataset
#    ↓
# PyTorch tensors
#    ↓
# DataLoader
#    ↓
# CNN training

import torch

from torch.utils.data import Dataset

import torchvision.transforms as transforms


# ==========================================================
# HYBRID DATASET
# ==========================================================
class HybridDataset(Dataset):

    def __init__(
        self,
        images,
        features,
        labels
    ):

        self.images = images

        self.features = features

        self.labels = labels

        self.transform = transforms.Compose([

            transforms.ToTensor(),

        ])


    def __len__(self):

        return len(self.images)


    def __getitem__(self, idx):

        image = self.transform(
            self.images[idx]
        )

        feature = torch.tensor(
            self.features[idx],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, feature, label