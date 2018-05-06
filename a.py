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
    related = relater(relatee)
    print('related', related.shape)


class Relater(nn.Module):
    def forward(self, x):
        return x


grid = torch.rand(32, 64, 4, 4)
context = None
relater = Relater()
global_pool = None
relate(grid, context, relater, global_pool)

print('-' * 80)

grid = torch.rand(32, 64, 4, 4)
context = torch.rand(32, 27)
relater = Relater()
global_pool = None
relate(grid, context, relater, global_pool)
