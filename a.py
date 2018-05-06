import numpy as np
import torch
from torch import nn


def relate(grid, context, relater, global_pool=None):
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

    print('cells', cells.shape)
    print('left', left.shape)
    print('right', right.shape)

    # Create grid x grid, concatenating context to each if given.
    if context is None:
        grid_x_grid = torch.cat([left, right], 0)
    else:
        context = context.unsqueeze(1)
        context = context.repeat(1, num_cells, 1)
        context = context.unsqueeze(2)
        context = context.repeat(1, 1, num_cells, 1)
        print('context', context.shape)
        grid_x_grid = torch.cat([left, right, context], 3)

    print('grid x grid', grid_x_grid.shape)

    # Reshape for feeding cell pairs to relater.
    relatee = grid_x_grid.view(batch_size * num_cells * num_cells, -1)
    print('relatee', relatee.shape)

    # Relate each pair of vectors (with optional context).
    x = relater(relatee)
    print('related', x.shape)

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
    print('global pooled', x.shape)
    return x


class Relater(nn.Module):
    def forward(self, x):
        return x


for global_pool in [None, 'avg', 'max']:
    print('=' * 80)
    print(global_pool)
    print()

    grid = torch.rand(32, 64, 4, 4)
    context = None
    relater = Relater()
    relate(grid, context, relater, global_pool)

    print('-' * 80)

    grid = torch.rand(32, 64, 4, 4)
    context = torch.rand(32, 27)
    relater = Relater()
    relate(grid, context, relater, global_pool)
