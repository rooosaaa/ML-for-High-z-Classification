"""
lyman_break_cnn.py — Lyman Break Classifier Model

This file defines the deep learning model used to classify galaxies into low-redshift,
high-redshift, or brown dwarf categories using JWST 7-band imaging. It includes a
custom ResNet-based architecture with spectral/spatial residual blocks and channel
attention to extract both spectral and spatial features from input cutouts.

Author: Rosa Roberts (rosa.roberts@student.manchester.ac.uk)
- Jodrell Bank Centre for Astrophysics, University of Manchester, Manchester, M13 9PL, UK.
Date: 04.08.2025
"""


# ========== IMPORTS ==========


# Import PyTorch
import torch
from torch import nn
import torch.nn.functional as F

# Check PyTorch environment
print(f"PyTorch version: {torch.__version__}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(device)}")


# JWST data specifics
field = 'JADES-DR3-GS-South'
filters = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']


# ========== MODEL COMPONENTS ==========


class SpectralResidualBlock(nn.Module):
    """Residual block for learning inter-band (spectral) relationships using 1×1 convolutions"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.1)  # Spatial dropout for regularization
        
    def forward(self, x):
        residual = x
        
        # First conv + bn + relu
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        # Second conv + bn (no relu yet)
        out = self.bn2(self.conv2(out))
        
        # Add skip connection and apply final activation
        out += residual
        return F.relu(out)

class SpatialResidualBlock(nn.Module):
    """Residual block for spatial feature extraction with 3×3 convolutions"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)
        
        # Skip connection (with downsampling if needed)
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential
            (
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class ChannelAttention(nn.Module):
    """Channel attention mechanism to emphasise informative bands using SE module"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential
        (
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Both average and max pooling
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        # Generate attention weights
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention


# ========== MAIN CLASSIFIER ==========


class LymanBreakClassifier(nn.Module):
    """CNN classifier using spectral + spatial residual blocks and channel attention"""
    def __init__(self, n_bands=8, image_size=64, dropout_rate=0.3):
        super().__init__()
        
        # Initial band-wise processing
        self.initial_conv = nn.Sequential
        (
            nn.Conv2d(n_bands, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Spectral feature learning
        self.spectral_blocks = nn.Sequential
        (
            SpectralResidualBlock(32),
            SpectralResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SpectralResidualBlock(64),
            SpectralResidualBlock(64),
        )
        
        # Channel weighting
        self.channel_attention = ChannelAttention(64)
        
        # Spatial processing
        self.spatial_blocks = nn.Sequential
        (
            SpatialResidualBlock(64, 64, stride=1),
            nn.MaxPool2d(2),  # 64 -> 32
            SpatialResidualBlock(64, 128, stride=1),
            nn.MaxPool2d(2),  # 32 -> 16
            SpatialResidualBlock(128, 256, stride=1),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        
        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential
        (
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 3)   # 3 classes: low-z, high-z, dwarf
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for convs and normals"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.initial_conv(x)
        x = self.spectral_blocks(x)
        x = self.channel_attention(x)
        x = self.spatial_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Return intermediate features for inspection/visualisation"""

        features = {}
        x = self.initial_conv(x)
        features['initial_spectral'] = x
        x = self.spectral_blocks(x)
        features['spectral_features'] = x
        x = self.channel_attention(x)
        features['attention_features'] = x
        x = self.spatial_blocks(x)
        features['spatial_features'] = x
        
        return features


# ========== HELPER FUNCTION TO CREATE MODEL ==========


# Example usage and model summary
def create_model(n_bands=7, image_size=64):
    """Instantiate model and print parameter count"""

    model = LymanBreakClassifier(n_bands=n_bands)
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


# ========== LOSS FUNCTION & OPTIMISER SETUP ==========


# Instantiate model
model = create_model(n_bands=7).to(device)
# Use class-balanced loss (adjust weights as needed)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]).to(device))
optimiser = torch.optim.Adam(model.parameters(), lr=0.0005)
    

