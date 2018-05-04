from argparse import ArgumentParser
import imageio
import numpy as np
from tqdm import tqdm


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--in', type=str, required=True)
    a.add_argument('--out_train_frames', type=str, required=True)
    a.add_argument('--out_train_indices', type=str, required=True)
    a.add_argument('--out_val_frames', type=str, required=True)
    a.add_argument('--out_val_indices', type=str, required=True)
    a.add_argument('--frame_count', type=int, default=4)
    a.add_argument('--frame_skip', type=int, default=8)
    a.add_argument('--val_frac', type=float, default=0.2)
    a.add_argument('--keep_frac', type=float, default=0.5)
    return a.parse_args()


def extract_sample(vid, begin_index, sample_shape, frame_count, frame_skip):
    x = np.zeros(sample_shape, 'uint8')
    for i in range(frame_count):
        index = begin_index + i * frame_skip
        x[i] = vid.get_data(index)
    return x


def process_split(vid, indices, out_frames_fn, out_indices_fn, sample_shape,
                  frame_count, frame_skip):
    x_shape = (len(indices),) + sample_shape
    x = np.zeros(x_shape, 'uint8')
    for i, index in enumerate(tqdm(indices)):
        x[i] = extract_sample(vid, index, sample_shape, frame_count, frame_skip)
    x.tofile(out_frames_fn)

    indices = np.array(indices, 'int32')
    indices.tofile(out_indices_fn)


def run(flags):
    tqdm.monitor_interval = 0

    vid = imageio.get_reader(getattr(flags, 'in'), 'ffmpeg')
    num_samples = len(vid) - flags.frame_count * flags.frame_skip + 1

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
    sample_shape = (flags.frame_count,) + im.shape
    print('Sample shape: %s (%d bytes)' % (sample_shape, np.prod(sample_shape)))

    print('Processing training split...')
    process_split(vid, train_indices, flags.out_train_frames,
                  flags.out_train_indices, sample_shape, flags.frame_count,
                  flags.frame_skip)

    print('Processing validation split...')
    process_split(vid, val_indices, flags.out_val_frames, flags.out_val_indices,
                  sample_shape, flags.frame_count, flags.frame_skip)


if __name__ == '__main__':
    run(parse_flags())
