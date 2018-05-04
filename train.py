from argparse import ArgumentParser
import numpy as np


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_train_frames', type=str,
                   default='data/train_128_train_frames.npy')
    a.add_argument('--in_train_indices', type=str,
                   default='data/train_128_train_indices.npy')
    a.add_argument('--in_val_frames', type=str,
                   default='data/train_128_val_frames.npy')
    a.add_argument('--in_val_indices', type=str,
                   default='data/train_128_val_indices.npy')
    a.add_argument('--sample_shape', type=str, default='4,128,128,3')
    a.add_argument('--frame_count', type=int, default=4)
    a.add_argument('--frame_skip', type=int, default=8)
    return a.parse_args()


def run(flags):
    sample_shape = tuple(map(int, flags.sample_shape.split(',')))

    x = np.fromfile(flags.in_train_frames, 'uint8')
    x = x.reshape(((-1,) + sample_shape))
    print('Training split: %s (%d bytes)' % (x.shape, np.prod(x.shape)))

    indices = np.fromfile(flags.in_train_indices, 'int32')
    assert len(indices) == len(x)

    x = np.fromfile(flags.in_val_frames, 'uint8')
    x = x.reshape(((-1,) + sample_shape))
    print('Validation split: %s (%d bytes)' % (x.shape, np.prod(x.shape)))

    indices = np.fromfile(flags.in_val_indices, 'int32')
    assert len(indices) == len(x)


if __name__ == '__main__':
    run(parse_flags())
