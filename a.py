import numpy as np
import torch
from torch import nn as ptnn
from torchplus import nn
from torchplus.nn.internal import Keyword


def api_relate(grid, context, relater, global_pool=None):
    # Get shapes.
    batch_size, num_grid_channels = grid.shape[:2]
    spatial_shape = grid.shape[2:]
    num_cells = int(np.prod(spatial_shape))

    # Flatten `grid` shapewise and put the channels dimension last.
    cells = grid.view(batch_size, num_grid_channels, num_cells)
    cells = cells.permute(0, 2, 1)

    # Repeat for concatenation.
    left = cells.unsqueeze(1)
    left = left.repeat(1, num_cells, 1, 1)
    right = cells.unsqueeze(2)
    right = right.repeat(1, 1, num_cells, 1)

    # Create grid x grid, concatenating context to each if given.
    if context is None:
        grid_x_grid = torch.cat([left, right], 0)
    else:
        context = context.unsqueeze(1)
        context = context.repeat(1, num_cells, 1)
        context = context.unsqueeze(2)
        context = context.repeat(1, 1, num_cells, 1)
        grid_x_grid = torch.cat([left, right, context], 3)

    # Reshape for feeding cell pairs to relater.
    x = grid_x_grid.view(batch_size * num_cells * num_cells, -1)

    # Relate each pair of vectors (with optional context).
    if relater:
        x = relater(x)

    if global_pool is None:
        # The idea: for each cell (depth x height x width), sum all the vectors
        # that involve it, preserving the input's spatial shape, allowing us to
        # plug this in as a layer and continue convolving or whatnot afterward.

        # Break it down into the actual dimensions.
        true_shape = (batch_size,) + spatial_shape + spatial_shape + (-1,)
        x = x.view(*true_shape)

        # Permute the depth, height, and width dimensions next to each other.
        permute_axes = [0, len(true_shape) - 1]
        for i in range(len(spatial_shape)):
            permute_axes.append(i + 1)  # Left side.
            permute_axes.append(i + 1 + len(spatial_shape))  # Right side.
        x = x.permute(*permute_axes)

        # Reduce over depth, height, and width.
        for i in reversed(range(len(spatial_shape))):
            x = x.sum((i + 1) * 2)
    elif global_pool == 'avg':
        # Reduce the output vectors to a single channel-wise vector.
        shape = batch_size, num_cells * num_cells, -1
        x = x.reshape(*shape)
        x = x.mean(1)
    elif global_pool == 'max':
        # Reduce the output vectors to a single channel-wise vector.
        shape = batch_size, num_cells * num_cells, -1
        x = x.reshape(*shape)
        x = x.max(1)[0]
    else:
        assert False
    return x


class DefaultRelater(ptnn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        block = lambda in_dim, out_dim: \
            nn.Linear(in_dim, out_dim) + nn.BatchNorm1d(out_dim) + nn.ReLU + \
            nn.Dropout

        mid_dim = (in_dim + out_dim) // 2
        one = block(in_dim, mid_dim)
        two = block(mid_dim, out_dim)

        self.seq = one + two

    def forward(self, x):
        return self.seq(x)


class Relate(ptnn.Module):
    def __init__(self, in_channels, out_channels, global_pool=None):
        super().__init__()
        assert global_pool in {None, 'avg', 'max'}
        self.relater = DefaultRelater(in_channels * 2, out_channels)
        self.global_pool = global_pool

    def forward(self, x):
        return api_relate(x, None, self.relater, self.global_pool)


Relate = Keyword(Relate)


class RelateWith(ptnn.Module):
    def __init__(self, in_channels, out_channels, context_dim,
                 global_pool=None):
        assert global_pool in {None, 'avg', 'max'}
        self.relater = DefaultRelater(
            in_channels * 2 + context_dim, out_channels)
        self.global_pool = global_pool


class ConvModel(ptnn.Module):
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

    def forward(self, x):
        return self.seq(x)


class RelModel(ptnn.Module):
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
        conv = c1 + c2 + c3 + c4 + c5

        # Relational blocks:
        r1 = Relate(8, 8)
        r2 = Relate(8, 8)
        relational = r1 + r2

        # Fully-connected blocks:
        d1 = nn.Linear(4 * 4 * n, n) + nn.BatchNorm1d(n) + nn.Dropout + nn.ReLU
        d2 = nn.Linear(n, 1) + nn.ReLU
        dense = nn.Flatten.instance() + d1 + d2

        self.seq = conv + relational + dense

    def forward(self, x):
        return self.seq(x)


for global_pool in [None, 'avg', 'max']:
    print('=' * 80)
    print(global_pool)
    print()

    model = ConvModel()

    x = torch.rand(32, 3, 6, 128, 128)
    y = model(x)
    print(x.shape, '->', y.shape)

    model = RelModel()

    x = torch.rand(32, 3, 6, 128, 128)
    y = model(x)
    print(x.shape, '->', y.shape)
