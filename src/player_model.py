"""A compact spatial ResNet encoder for Go player style; no classification head."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection preserve the board resolution."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        """Add residual features and apply ReLU."""
        return F.relu(x + self.layers(x))


class PlayerEncoder(nn.Module):
    """Map (B,17,19,19) histories to unit-length style embeddings."""

    def __init__(self, in_channels=17, channels=64, num_blocks=8,
                 embedding_dim=128, pool_size=3):
        super().__init__()
        if min(in_channels, channels, num_blocks, embedding_dim, pool_size) < 1:
            raise ValueError('All model dimensions must be positive')
        self.in_channels = in_channels
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(num_blocks)))
        # Coarse spatial pooling retains region preferences with a small projection.
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.projection = nn.Linear(channels * pool_size * pool_size, embedding_dim)

    def forward(self, x):
        """Encode board history and L2-normalize along the embedding dimension."""
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, 19, 19):
            raise ValueError(f'Expected (B,{self.in_channels},19,19); got {tuple(x.shape)}')
        x = self.blocks(self.stem(x))
        x = self.projection(torch.flatten(self.pool(x), 1))
        return F.normalize(x, p=2, dim=1)
