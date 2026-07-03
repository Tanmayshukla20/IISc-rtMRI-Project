# combines image learning and physiological learning into one biomedical AI classifier.

# ROI Image
#      ↓
# ResNet CNN
#      ↓
# 512 image features

# Physiological Features
#      ↓
# Feature Branch
#      ↓
# 32 physiological features

# Both Combined
#      ↓
# Fusion Classifier
#      ↓
# Voiced / Unvoiced

import torch
import torch.nn as nn

import torchvision.models as models


# ==========================================================
# HYBRID CNN MODEL
# ==========================================================
class HybridCNN(nn.Module):

    def __init__(self, num_features):

        super(HybridCNN, self).__init__()

        # ==================================================
        # CNN BACKBONE
        # ==================================================
        self.cnn = models.resnet18(
            weights=None
        )

        self.cnn.fc = nn.Identity()


        # ==================================================
        # PHYSIOLOGICAL FEATURE BRANCH
        # ==================================================
        self.feature_branch = nn.Sequential(

            nn.Linear(num_features, 32),

            nn.ReLU(),

            nn.Dropout(0.2)

        )


        # ==================================================
        # FEATURE FUSION CLASSIFIER
        # ==================================================
        self.classifier = nn.Sequential(

            nn.Linear(512 + 32, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 2)

        )


    # ======================================================
    # FORWARD PASS
    # ======================================================
    def forward(
        self,
        image,
        features
    ):

        # CNN IMAGE FEATURES
        cnn_features = self.cnn(image)

        # PHYSIOLOGICAL FEATURES
        phys_features = self.feature_branch(
            features
        )

        # FEATURE FUSION
        fused = torch.cat([

            cnn_features,

            phys_features

        ], dim=1)

        # FINAL CLASSIFICATION
        output = self.classifier(
            fused
        )

        return output
    