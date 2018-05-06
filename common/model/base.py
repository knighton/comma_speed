import numpy as np
import os
import re
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


tqdm.monitor_interval = 0


class BaseModel(nn.Module):
    def transform_clips(self, x):
        x = x.astype('float32')
        x /= 127.5
        x -= 1
        return torch.from_numpy(x).cuda()

    def transform_speeds(self, y):
        return torch.from_numpy(y).cuda()

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
        total = dataset.batches_per_epoch(batch_size)
        each_batch = tqdm(each_batch, total=total, leave=False)
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
            else:
                self.eval()
                loss = self.val_on_batch(clips, speeds)
                val_losses.append(loss)
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
            batch_size=32, chk_dir=None):
        print('Fit from epoch %d -> %d' % (begin_epoch, end_epoch))
        for epoch in range(begin_epoch, end_epoch):
            train_loss, val_loss = self.fit_on_epoch(
                dataset, optimizer, batch_size)
            if chk_dir:
                self.save(chk_dir, epoch, train_loss, val_loss)
            print('epoch %d: %.4f %.4f' % (epoch, train_loss, val_loss))
