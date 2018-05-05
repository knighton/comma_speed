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
    a.add_argument('--clip_len', type=int, default=5)
    a.add_argument('--clip_stride', type=int, default=10)
    a.add_argument('--val_frac', type=float, default=0.2)
    a.add_argument('--keep_frac', type=float, default=0.5)
    return a.parse_args()


def extract_clip(vid, begin_index, clip_shape, clip_len, clip_stride):
    x = np.zeros(clip_shape, 'uint8')
    for i in range(clip_len):
        index = begin_index + i * clip_stride
        x[i] = vid.get_data(index)
    return x


def process_split(vid, indices, out_frames_fn, out_indices_fn, clip_shape,
                  clip_len, clip_stride):
    x_shape = (len(indices),) + clip_shape
    x = np.zeros(x_shape, 'uint8')
    for i, index in enumerate(tqdm(indices)):
        x[i] = extract_clip(vid, index, clip_shape, clip_len, clip_stride)
    x.tofile(out_frames_fn)

    indices = np.array(indices, 'int32')
    indices.tofile(out_indices_fn)


def run(flags):
    tqdm.monitor_interval = 0

    vid = imageio.get_reader(getattr(flags, 'in'), 'ffmpeg')
    num_samples = len(vid) - flags.clip_len * flags.clip_stride + 1

    is_sample_train = flags.val_frac < np.random.uniform(0, 1, num_samples)

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

    im = vid.get_data(0)
    clip_shape = (flags.clip_len,) + im.shape
    print('Sample shape: %s (%d bytes)' % (clip_shape, np.prod(clip_shape)))

    print('Processing training split...')
    process_split(vid, train_indices, flags.out_train_clips,
                  flags.out_train_indices, clip_shape, flags.clip_len,
                  flags.clip_stride)

    print('Processing validation split...')
    process_split(vid, val_indices, flags.out_val_clips, flags.out_val_indices,
                  clip_shape, flags.clip_len, flags.clip_stride)


if __name__ == '__main__':
    run(parse_flags())
