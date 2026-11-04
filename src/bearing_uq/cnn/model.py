"""WDCNN: a 1D-CNN for raw vibration signals with a wide first-layer kernel.

Architecture of Zhang et al. (2017), "A new deep learning model for fault diagnosis with good anti-noise and
domain adaptation ability on raw vibration signals", Sensors 17(2), 425. The first convolution (kernel 64,
stride 16) acts as a learned filter bank; four small convolutions follow. Input: 4,096-sample segments.
"""

import torch
from torch import nn

SEGMENT = 4096


def _block(c_in, c_out, kernel, stride=1, padding=1):
    return [nn.Conv1d(c_in, c_out, kernel, stride, padding), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.MaxPool1d(2)]


class WDCNN(nn.Module):
    def __init__(self, n_classes=3, segment=SEGMENT):
        super().__init__()
        self.features = nn.Sequential(
            *_block(1, 16, 64, stride=16, padding=24),
            *_block(16, 32, 3), *_block(32, 64, 3), *_block(64, 64, 3),
            *_block(64, 64, 3, padding=0),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, segment)).numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                        nn.Linear(100, n_classes))

    def forward(self, x):  # x: (batch, segment)
        return self.classifier(self.features(x.unsqueeze(1)))
