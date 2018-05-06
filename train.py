from argparse import ArgumentParser
import numpy as np
import os
import torch
from torch.optim import SGD

from common.dataset import Dataset, RamSplit
from common.model.conv import ConvModel


def parse_flags():
    a = ArgumentParser()

    # Inputs.
    a.add_argument('--in_train_clips', type=str,
                   default='data/train_clips.npy')
    a.add_argument('--in_train_indices', type=str,
                   default='data/train_indices.npy')
    a.add_argument('--in_val_clips', type=str,
                   default='data/val_clips.npy')
    a.add_argument('--in_val_indices', type=str,
                   default='data/val_indices.npy')
    a.add_argument('--in_speeds', type=str, default='data/train.txt')

    # Outputs.
    a.add_argument('--out_model', type=str, required=True)

    # Data.
    a.add_argument('--clip_len', type=int, default=6)
    a.add_argument('--frame_height', type=int, default=128)
    a.add_argument('--frame_width', type=int, default=128)

    # Model.
    a.add_argument('--dim', type=int, required=True)

    # Training.
    a.add_argument('--lr', type=float, default=0.001)
    a.add_argument('--momentum', type=float, default=0.9)
    a.add_argument('--end_epoch', type=int, default=1000)
    a.add_argument('--batch_size', type=int, default=32)

    return a.parse_args()


def load_split(clips_fn, indices_fn, clip_len, frame_shape, speeds):
    x = np.fromfile(clips_fn, 'uint8')
    x_shape = (-1, clip_len) + frame_shape
    x = x.reshape(x_shape)
    x = x.transpose([0, 2, 1, 3, 4])
    indices = np.fromfile(indices_fn, 'int32')
    assert len(indices) == len(x)
    y = speeds[indices]
    y = np.expand_dims(y, 1)
    return RamSplit(x, y)


def run(flags):
    frame_shape = 3, flags.frame_height, flags.frame_width
    speeds = np.array(list(map(float, open(flags.in_speeds))), 'float32')
    train = load_split(flags.in_train_clips, flags.in_train_indices,
                       flags.clip_len, frame_shape, speeds)
    val = load_split(flags.in_val_clips, flags.in_val_indices, flags.clip_len,
                     frame_shape, speeds)
    dataset = Dataset(train, val)
    model = ConvModel(flags.dim).cuda()
    optimizer = SGD(model.parameters(), lr=flags.lr, momentum=flags.momentum)
    begin_epoch = model.maybe_load_last_epoch(flags.out_model)
    model.fit(dataset, optimizer, begin_epoch=begin_epoch,
              end_epoch=flags.end_epoch, batch_size=flags.batch_size,
              chk_dir=flags.out_model)


if __name__ == '__main__':
    run(parse_flags())
