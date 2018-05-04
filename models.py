import numpy as np
import sys
import torch
from torch import nn as ptnn
from torch.nn import functional as F
from torchplus import nn
from tqdm import tqdm


tqdm.monitor_interval = 0


class SpeedPredictor(ptnn.Module):
    def transform_clips(self, x):
        # Input shape: (N, T, H, W, C).
        #     example: (32, 4, 128, 128, 3).
        x = x.transpose([0, 4, 2, 3, 1])
        x = x.astype('float32')
        x /= 127.5
        x -= 1
        return torch.from_numpy(x)

    def transform_speeds(self, x):
        x = np.expand_dims(x, 1)
        return torch.from_numpy(x)

    def train_on_batch(self, optimizer, clips, true_speeds):
        optimizer.zero_grad()
        pred_speeds = self.forward(clips)
        loss = F.mse_loss(pred_speeds, true_speeds)
        loss.backward()
        optimizer.step()
        return np.asscalar(loss.detach().cpu().numpy())

    def val_on_batch(self, clips, true_speeds):
        pred_speeds = self.forward(clips)
        loss = F.mse_loss(pred_speeds, true_speeds)
        return np.asscalar(loss.detach().cpu().numpy())

    def fit_on_epoch(self, dataset, optimizer, batch_size):
        each_batch = dataset.each_batch(batch_size)
        #total = dataset.batches_per_epoch(batch_size)
        #each_batch = tqdm(each_batch, total=total)
        train_losses = []
        val_losses = []
        for is_training, xx, yy in each_batch:
            clips, = xx
            clips = self.transform_clips(clips)
            speeds, = yy
            speeds = self.transform_speeds(speeds)
            if is_training:
                self.train()
                loss = self.train_on_batch(optimizer, clips, speeds)
                train_losses.append(loss)
                sys.stdout.write('T %.4f\n' % loss)
                sys.stdout.flush()
            else:
                self.eval()
                loss = self.val_on_batch(clips, speeds)
                val_losses.append(loss)
                sys.stdout.write('V %.4f\n' % loss)
                sys.stdout.flush()
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        return train_loss, val_loss

    def fit(self, dataset, optimizer, num_epochs, batch_size):
        for epoch in range(num_epochs):
            train_loss, val_loss = self.fit_on_epoch(
                dataset, optimizer, batch_size)
            print('epoch %d: %.4f %.4f' % (epoch, train_loss, val_loss))


class Model(SpeedPredictor):
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
        d2 = nn.Linear(n, 1)

        self.seq = c1 + c2 + c3 + c4 + c5 + nn.Flatten + d1 + d2

    def forward(self, x):
        return self.seq(x)
