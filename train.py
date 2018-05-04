from argparse import ArgumentParser
import numpy as np


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in_train', type=str, default='data/train_128_train.npy')
    a.add_argument('--in_val', type=str, default='data/train_128_val.npy')
    a.add_argument('--sample_shape', type=str, default='4,128,128,3')
    a.add_argument('--frame_count', type=int, default=4)
    a.add_argument('--frame_skip', type=int, default=8)
    return a.parse_args()


def run(flags):
    sample_shape = tuple(map(int, flags.sample_shape.split(',')))

    x = np.fromfile(flags.in_train, 'uint8')
    x = x.reshape(((-1,) + sample_shape))
    print('Training split: %s (%d bytes)' % (x.shape, np.prod(x.shape)))

    x = np.fromfile(flags.in_val, 'uint8')
    x = x.reshape(((-1,) + sample_shape))
    print('Validation split: %s (%d bytes)' % (x.shape, np.prod(x.shape)))


if __name__ == '__main__':
    run(parse_flags())
