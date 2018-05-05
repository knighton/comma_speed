from argparse import ArgumentParser
import imageio
import numpy as np
from tqdm import tqdm


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in', type=str, default='data/test_128.mp4')
    a.add_argument('--out', type=str, default='data/test_clips.npy')
    a.add_argument('--clip_len', type=int, default=6)
    a.add_argument('--clip_stride', type=int, default=4)
    return a.parse_args()


def get_frame(vid, index):
    x = vid.get_data(index)
    x = x.transpose([2, 0, 1])
    return x


def get_frame_or_blank(vid, index, frame_shape):
    if index < 0:
        im = np.zeros(frame_shape, 'uint8')
    else:
        im = get_frame(vid, index)
    return im


def extract_clip(vid, now, frame_shape, clip_len, clip_stride):
    clip_shape = (clip_len,) + frame_shape
    clip = np.zeros(clip_shape, 'uint8')
    for i in range(clip_len):
        frame_index = now + (i - clip_len + 1) * clip_stride
        clip[i] = get_frame_or_blank(vid, frame_index, frame_shape)
    return clip


def load_split(vid, clip_len, clip_stride, frame_shape):
    x_shape = (len(vid), clip_len) + frame_shape
    x = np.zeros(x_shape, 'uint8')
    for now in tqdm(range(len(vid))):
        x[now] = extract_clip(vid, now, frame_shape, clip_len, clip_stride)
    return x


def run(flags):
    tqdm.monitor_interval = 0
    vid = imageio.get_reader(getattr(flags, 'in'), 'ffmpeg')
    im = get_frame(vid, 0)
    x = load_split(vid, flags.clip_len, flags.clip_stride, im.shape)
    x.tofile(flags.out)


if __name__ == '__main__':
    run(parse_flags())
