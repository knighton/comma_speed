import numpy as np
import os
import re
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

    def fit_on_epoch(self, dataset, optimizer, batch_size, batches_per_log=-1):
        each_batch = dataset.each_batch(batch_size)
        #total = dataset.batches_per_epoch(batch_size)
        #each_batch = tqdm(each_batch, total=total)
        train_losses = []
        val_losses = []
        for batch_index, (is_training, xx, yy) in enumerate(each_batch):
            clips, = xx
            clips = self.transform_clips(clips)
            speeds, = yy
            speeds = self.transform_speeds(speeds)
            if is_training:
                self.train()
                loss = self.train_on_batch(optimizer, clips, speeds)
                train_losses.append(loss)
                if batches_per_log != -1 and not batch_index % batches_per_log:
                    sys.stdout.write('T %.4f\n' % loss)
                    sys.stdout.flush()
            else:
                self.eval()
                loss = self.val_on_batch(clips, speeds)
                val_losses.append(loss)
                if batches_per_log != -1 and not batch_index % batches_per_log:
                    sys.stdout.write('%sV %.4f\n' % (' ' * 20, loss))
                    sys.stdout.flush()
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        return train_loss, val_loss

    def save(self, chk_dir, epoch, train_loss, val_loss):
        out = '%04d_%07d_%07d.bin' % \
            (epoch, int(train_loss * 1000), int(val_loss * 1000))
        out = os.path.join(chk_dir, out)
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        torch.save(self.state_dict(), out)

    def convert(self):
        # Call convert() on any 'torchplus' models.  Is hack.
        pass

    def load(self, filename):
        print('Loading model checkpoint: %s' % filename)
        self.convert()
        self.load_state_dict(torch.load(filename))

    def maybe_load_last_epoch(self, chk_dir):
        print('Loading model checkpoints dir: %s' % chk_dir)
        pat = '[0-9]+_[0-9]+_[0-9]+.bin'
        if not os.path.isdir(chk_dir):
            print('Checkpoint dir does not exist')
            return 0
        ff = os.listdir(chk_dir)
        ff = filter(lambda f: re.match(pat, f), ff)
        ff = list(map(lambda f: os.path.join(chk_dir, f), ff))
        if not ff:
            print('No checkpoints found')
            return 0
        ff.sort()
        f = ff[-1]
        self.load(f)
        x = f.rindex(os.path.sep)
        f = f[x + 1:]
        x = f.index('_')
        last_epoch = int(f[:x])
        print('Loaded epoch %d' % last_epoch)
        return last_epoch + 1

    def fit(self, dataset, optimizer, begin_epoch=0, end_epoch=10,
            batch_size=32, batches_per_log=-1, chk_dir=None):
        print('Fit from epoch %d -> %d' % (begin_epoch, end_epoch))
        for epoch in range(begin_epoch, end_epoch):
            train_loss, val_loss = self.fit_on_epoch(
                dataset, optimizer, batch_size, batches_per_log)
            if chk_dir:
                self.save(chk_dir, epoch, train_loss, val_loss)
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

    def convert(self):
        self.seq.convert()

    def forward(self, x):
        return self.seq(x)
