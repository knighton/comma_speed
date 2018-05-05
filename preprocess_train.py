from argparse import ArgumentParser
import imageio
import numpy as np
from tqdm import tqdm


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in', type=str, required=True)
    a.add_argument('--out_train_clips', type=str, required=True)
    a.add_argument('--out_train_indices', type=str, required=True)
    a.add_argument('--out_val_clips', type=str, required=True)
    a.add_argument('--out_val_indices', type=str, required=True)
    a.add_argument('--clip_len', type=int, default=6)
    a.add_argument('--clip_stride', type=int, default=4)
    a.add_argument('--val_frac', type=float, default=0.2)
    a.add_argument('--keep_frac', type=float, default=0.5)
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


def process_split(vid, indices, out_frames_fn, out_indices_fn, frame_shape,
                  clip_len, clip_stride):
    x_shape = (len(indices), clip_len) + frame_shape
    x = np.zeros(x_shape, 'uint8')
    for i, now in enumerate(tqdm(indices)):
        x[i] = extract_clip(vid, now, frame_shape, clip_len, clip_stride)
    x.tofile(out_frames_fn)

    indices = np.array(indices, 'int32')
    indices.tofile(out_indices_fn)


def run(flags):
    tqdm.monitor_interval = 0

    vid = imageio.get_reader(getattr(flags, 'in'), 'ffmpeg')
    is_sample_train = flags.val_frac < np.random.uniform(0, 1, len(vid))

    train_indices = []
    val_indices = []
    for i, is_training in enumerate(is_sample_train):
        if is_training:
            train_indices.append(i)
        else:
            val_indices.append(i)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    print('Sample counts: %d train, %d val' %
          (len(train_indices), len(val_indices)))

    train_indices = train_indices[:int(len(train_indices) * flags.keep_frac)]
    val_indices = val_indices[:int(len(val_indices) * flags.keep_frac)]
    print('Kept sample counts: %d train, %d val' %
          (len(train_indices), len(val_indices)))

    im = get_frame(vid, 0)
    clip_shape = (flags.clip_len,) + im.shape
    print('Sample shape: %s (%d bytes)' % (clip_shape, np.prod(clip_shape)))

    print('Processing training split...')
    process_split(vid, train_indices, flags.out_train_clips,
                  flags.out_train_indices, im.shape, flags.clip_len,
                  flags.clip_stride)

    print('Processing validation split...')
    process_split(vid, val_indices, flags.out_val_clips, flags.out_val_indices,
                  im.shape, flags.clip_len, flags.clip_stride)


if __name__ == '__main__':
    run(parse_flags())
