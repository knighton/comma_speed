import numpy as np
import torch
from torch import nn


def relate(grid, context, relater, global_pool=None):
    # Get shapes.
    batch_size, num_grid_channels = grid.shape[:2]
    spatial_shape = grid.shape[2:]
    num_cells = np.prod(spatial_shape)

    # Flatten `grid` shapewise and put the channels dimension last.
    cells = grid.view(batch_size, num_grid_channels, num_cells)
    cells = cells.permute(0, 2, 1)

    # Repeat for concatenation.
    left = cells.unsqueeze(1)
    left = left.repeat(1, num_cells, 1, 1)
    right = cells.unsqueeze(2)
    right = right.repeat(1, 1, num_cells, 1)

    print(cells.shape, left.shape, right.shape)


class Relater(nn.Module):
    def forward(self, x):
        return x


grid = torch.rand(32, 64, 4, 4)
context = None
relater = Relater()
global_pool = None
relate(grid, context, relater, global_pool)
