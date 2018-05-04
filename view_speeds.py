from argparse import ArgumentParser
import numpy as np


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in', type=str, required=True)
    a.add_argument('--max_bar_len', type=int, default=80)
    return a.parse_args()


def run(flags):
    xx = np.array(list(map(float, open(getattr(flags, 'in')))), 'float32')
    nn = xx.astype('int16')  # Fuck the space shuttle
    buckets = np.zeros(nn.max() + 1, 'int64')
    for n in nn:
        buckets[n] += 1
    max_bucket = buckets.max()
    for i, bucket in enumerate(buckets):
        bar_len = int(bucket / max_bucket * flags.max_bar_len)
        bar = '-' * bar_len
        print('%2d %7d %s' % (i, bucket, bar))


if __name__ == '__main__':
    run(parse_flags())
