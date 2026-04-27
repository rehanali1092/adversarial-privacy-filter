# FGSM and PGD attack code - Day 2
"""
model.py
Author: Rehan Ali
Face Recognition Model using FaceNet
"""

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceRecognitionModel(nn.Module):
    """
    Face Recognition Model
    Base: FaceNet (InceptionResnetV1) pretrained on VGGFace2
    Head: Custom classifier for LFW identities
    """

    def __init__(self, num_classes=62):
        super(FaceRecognitionModel, self).__init__()

        # Pretrained FaceNet feature extractor
        self.feature_extractor = InceptionResnetV1(
            pretrained='vggface2',
            classify=False
        )

        # Freeze all FaceNet layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze last block for fine tuning
        for param in self.feature_extractor.block8.parameters():
            param.requires_grad = True
        for param in self.feature_extractor.last_linear.parameters():
            param.requires_grad = True
        for param in self.feature_extractor.last_bn.parameters():
            param.requires_grad = True

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embeddings = self.feature_extractor(x)
        logits     = self.classifier(embeddings)
        return logits, embeddings
