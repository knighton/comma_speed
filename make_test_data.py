from argparse import ArgumentParser
import imageio
import numpy as np
from tqdm import tqdm


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in', type=str, default='data/test_128.mp4')
    a.add_argument('--out', type=str, default='data/test_128_x.npy')
    a.add_argument('--frame_count', type=int, default=4)
    a.add_argument('--frame_skip', type=int, default=8)
    return a.parse_args()


def get_frame_or_blank(vid, index, frame_shape):
    if index < 0:
        im = np.zeros(frame_shape, 'uint8')
    else:
        im = vid.get_data(index)
    return im


def extract_context_for(vid, frame_index, frame_shape, frame_count,
                        frame_skip):
    x_shape = (frame_count,) + frame_shape
    x = np.zeros(x_shape, 'uint8')
    for i in range(frame_count):
        index = frame_index + (i - frame_count) * frame_skip  # Yeah...
        x[i] = get_frame_or_blank(vid, index, frame_shape)
    return x


def make_split(vid, frame_count, frame_skip):
    n = len(vid)
    im = vid.get_data(0)
    frame_shape = im.shape
    x_shape = (n, frame_count) + frame_shape
    x = np.zeros(x_shape, 'uint8')
    for i in tqdm(range(n), total=n):
        x[i] = extract_context_for(vid, i, frame_shape, frame_count, frame_skip)
    return x


def run(flags):
    tqdm.monitor_interval = 0
    vid = imageio.get_reader(getattr(flags, 'in'), 'ffmpeg')
    x = make_split(vid, flags.frame_count, flags.frame_skip)
    x.tofile(flags.out)


if __name__ == '__main__':
    run(parse_flags())
