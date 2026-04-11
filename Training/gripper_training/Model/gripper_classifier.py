import torch
import torch.nn as nn
from r3m import load_r3m
"""
USING R3M IN THE MODEL (EXTRACTING FEATURES FIRST IS MUCH FASTER):
class GripperClassifier(nn.Module):
    def __init__(self, proprio_dim=18, in_channels=6):
        super().__init__()
        self.image_feature = load_r3m("resnet18")
        for param in self.image_feature.parameters():
            param.requires_grad = False

        num_cams    = in_channels // 3
        feature_dim = 512 * num_cams   # 1024 for 2 cameras, 512 for 1

        self.projection = nn.Sequential(
            nn.Linear(feature_dim + proprio_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )

    def forward(self, current_image, proprio):

        num_cams = current_image.shape[1] // 3

        # Split cameras and encode each one separately through R3M
        cam_features = []
        for i in range(num_cams):
            cam = current_image[:, i*3:(i+1)*3, :, :]   # (B, 3, 224, 224)
            with torch.no_grad():
                feat = self.image_feature(cam).view(-1, 512)  # (B, 512)
            cam_features.append(feat)

        # Concatenate along feature dim: (B, 512 * num_cams)
        image_features = torch.cat(cam_features, dim=-1)

        combined = torch.cat([image_features, proprio], dim=-1)
        return self.projection(combined)

class GripperClassifier(nn.Module):
    def __init__(self, proprio_dim=18, num_cams=2, T=5):
        super().__init__()
        feature_dim = 512 * num_cams   # 1024 for 2 cameras
        proprio_total = proprio_dim*T #T is the number of timesteps before the current frame that it stores for prioprioception history

        self.projection = nn.Sequential(
            nn.Linear(feature_dim + proprio_total, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image_features, proprio):
        combined = torch.cat([image_features, proprio], dim=-1)
        return self.projection(combined)

    def predict_proba(self, current_image, proprio):
        logit = self.forward(current_image, proprio)
        return torch.sigmoid(logit)
"""




class GripperClassifier(nn.Module):
    def __init__(self, proprio_dim=18, num_cams=2):
        super().__init__()
        feature_dim = 512 * num_cams   # 1024 for 2 cameras
        proprio_total = proprio_dim #T is the number of timesteps before the current frame that it stores for prioprioception history

        self.projection = nn.Sequential(
            nn.Linear(feature_dim + proprio_total, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image_features, proprio):
        combined = torch.cat([image_features, proprio], dim=-1)
        return self.projection(combined)

    def predict_proba(self, current_image, proprio):
        logit = self.forward(current_image, proprio)
        return torch.sigmoid(logit)

