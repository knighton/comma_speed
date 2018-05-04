import imageio
import numpy as np


class Loader(object):
    def __init__(self, filename, batch_size=32, frames_per_sample=8,
                 val_frac=0.2):
        self.filename = filename
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.val_frac = val_frac

        self.vid = imageio.get_reader(filename, 'ffmpeg')
        self.num_samples = len(self.vid) - frames_per_sample + 1

        im = self.vid.get_data(0)
        self.sample_shape = (self.frames_per_sample,) + im.shape
        print('Sample shape: %s' % (self.sample_shape,))

        self.is_sample_train = \
            val_frac < np.random.uniform(0, 1, self.num_samples)

        train_indices = []
        val_indices = []
        for i, is_training in enumerate(self.is_sample_train):
            if is_training:
                train_indices.append(i)
            else:
                val_indices.append(i)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        self.train_batches = []
        for i in range(len(train_indices) // self.batch_size):
            a = i * self.batch_size
            z = (i + 1) * self.batch_size
            batch = train_indices[a:z]
            self.train_batches.append(batch)

        self.val_batches = []
        for i in range(len(val_indices) // self.batch_size):
            a = i * self.batch_size
            z = (i + 1) * self.batch_size
            batch = val_indices[a:z]
            self.val_batches.append(batch)
        print('Batches: %d train, %d val' %
              (len(self.train_batches), len(self.val_batches)))

    def get_sample(self, index):
        x = np.zeros(self.sample_shape, 'uint8')
        for i in range(self.frames_per_sample):
            x[i] = self.vid.get_data(index + i)
        return x


loader = Loader('data/train.mp4')
x = loader.get_sample(1337)
