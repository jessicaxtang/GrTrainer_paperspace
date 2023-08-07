import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

import copy


class PreProcess(nn.Module):
    """
    Preprocess the rgb and depth image with 50 residual layers
    """
    
    def __init__(self):
        super(PreProcess, self).__init__()
        self.rgb_features = resnet50(pretrained=True)
        self.d_features = resnet50(pretrained=True)

    def preprocess_patch(self, patch_rgb, patch_depth):
        # Resize the RGB patch to 224x224 using OpenCV
        resized_patch_rgb = cv2.resize(patch_rgb, (224, 224))
        
        # Rescale the depth image to range 0 to 255
        min_depth = np.nanmin(patch_depth)
        max_depth = np.nanmax(patch_depth)
        
        # Replace NaN values with zeros
        patch_depth = np.nan_to_num(patch_depth)
        
        # Rescale the depth values to the 0-255 range
        scaled_patch_depth = 255 * (patch_depth - min_depth) / (max_depth - min_depth)
        
        # Convert the depth image to uint8
        scaled_patch_depth = scaled_patch_depth.astype(np.uint8)
        
        return resized_patch_rgb, scaled_patch_depth
                    


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class mmGrPred(nn.Module):
    def __init__(self):
        super(mmGrPred, self).__init__()
        pretrained_resnet50 = resnet50(pretrained=True)
        self.rgb_features = copy.deepcopy(pretrained_resnet50.features[:3])
        self.d_features = copy.deepcopy(pretrained_resnet50.features[:3])
        self.features = nn.Sequential(
            # pool of square window of size=3, stride=2
            nn.AvgPool2d(kernel_size=3, stride=2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.grasp = nn.Sequential(
            nn.ConvTranspose2d(16, 5, kernel_size=11, stride=4, output_padding=1),
            nn.Tanh()
        )

        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=11, stride=4, output_padding=1),
            nn.Sigmoid()
        )

        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

        # xavier initialization for combined feature extractor
        for m in self.features.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        for m in self.grasp.modules():
            if isinstance(m, (nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
        for m in self.confidence.modules():
            if isinstance(m, (nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.rgb_features(rgb)
        d = self.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        x = self.features(x)
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)

        return out

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True
        
        for param in self.d_features.parameters():
            param.requires_grad = True