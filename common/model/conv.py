from torchplus import nn

from .base import BaseModel


class ConvModel(BaseModel):
    def __init__(self, n=8):
        super().__init__()

        # Conv blocks:

        # 128 -> 64.
        c1 = nn.Conv3d(3, n, 5, 2, 2) + nn.BatchNorm3d(n) + nn.ReLU

        # 64 -> 32.
        c2 = nn.Conv3d(n, n, 5, 2, 2) + nn.BatchNorm3d(n) + nn.ReLU

        # 32 -> 16.
        c3 = nn.Conv3d(n, n, 5, 2, 2) + nn.BatchNorm3d(n) + nn.ReLU

        # 16 -> 8.
        c4 = nn.Conv3d(n, n, 5, 2, 2) + nn.BatchNorm3d(n) + nn.ReLU + \
            nn.Dropout2d

        # 8 -> 4.
        c5 = nn.Conv3d(n, n, 5, 2, 2) + nn.BatchNorm3d(n) + nn.ReLU + \
            nn.Dropout2d

        # Fully-connected blocks:
        d1 = nn.Linear(4 * 4 * n, n) + nn.BatchNorm1d(n) + nn.Dropout + nn.ReLU
        d2 = nn.Linear(n, 1) + nn.ReLU

        self.seq = c1 + c2 + c3 + c4 + c5 + nn.Flatten + d1 + d2

    def convert(self):
        self.seq.convert()

    def forward(self, x):
        return self.seq(x)
