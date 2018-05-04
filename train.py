from argparse import ArgumentParser
import numpy as np
import os
import torch
from torch.optim import SGD

from common import Dataset, RamSplit
from models import Model


def parse_flags():
    a = ArgumentParser()

    # Inputs.
    a.add_argument('--in_train_frames', type=str,
                   default='data/train_128_train_frames.npy')
    a.add_argument('--in_train_indices', type=str,
                   default='data/train_128_train_indices.npy')
    a.add_argument('--in_val_frames', type=str,
                   default='data/train_128_val_frames.npy')
    a.add_argument('--in_val_indices', type=str,
                   default='data/train_128_val_indices.npy')
    a.add_argument('--in_speeds', type=str, default='data/train.txt')

    # Outputs.
    a.add_argument('--out_model', type=str, required=True)

    # Data.
    a.add_argument('--sample_shape', type=str, default='4,128,128,3')
    a.add_argument('--frame_count', type=int, default=4)
    a.add_argument('--frame_skip', type=int, default=8)

    # Model.
    a.add_argument('--dim', type=int, default=8)

    # Training.
    a.add_argument('--num_epochs', type=int, default=1000)
    a.add_argument('--batch_size', type=int, default=32)
    a.add_argument('--batches_per_log', type=int, default=4)

    return a.parse_args()


def load_split(frames_fn, indices_fn, sample_shape, speeds, speed_frame_offset):
    x = np.fromfile(frames_fn, 'uint8')
    x = x.reshape(((-1,) + sample_shape))
    print('Training split: %s (%d bytes)' % (x.shape, np.prod(x.shape)))
    indices = np.fromfile(indices_fn, 'int32')
    assert len(indices) == len(x)
    y = speeds[indices + speed_frame_offset]
    return RamSplit(x, y)


def run(flags):
    sample_shape = tuple(map(int, flags.sample_shape.split(',')))
    speeds = np.array(list(map(float, open(flags.in_speeds))), 'float32')
    speed_frame_offset = flags.frame_count + flags.frame_skip
    train = load_split(flags.in_train_frames, flags.in_train_indices,
                       sample_shape, speeds, speed_frame_offset)
    val = load_split(flags.in_val_frames, flags.in_val_indices, sample_shape,
                     speeds, speed_frame_offset)
    dataset = Dataset(train, val)
    model = Model(flags.dim)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.fit(dataset, optimizer, begin_epoch=0, end_epoch=flags.num_epochs,
              batch_size=flags.batch_size,
              batches_per_log=flags.batches_per_log, chk_dir=flags.out_model)


if __name__ == '__main__':
    run(parse_flags())
