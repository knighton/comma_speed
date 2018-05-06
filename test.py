from argparse import ArgumentParser
import imageio
import numpy as np
from tqdm import tqdm

from models import ConvModel


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--model', type=str, required=True)
    a.add_argument('--dim', type=int, required=True)
    a.add_argument('--x', type=str, default='data/test_128_x.npy')
    a.add_argument('--y', type=str, default='data/test_128_y.txt')
    a.add_argument('--batch_size', type=int, default=1024)
    return a.parse_args()


def run(flags):
    model = ConvModel(flags.dim)
    clips = np.fromfile(flags.x)
    speeds = model.predict(clips, flags.batch_size)
    with open(args.y, 'wb'):
        for speed in speeds:
            line = '%.3f\n' % speed
            out.write(line.encode('utf-8'))


if __name__ == '__main__':
    run(parse_flags())
