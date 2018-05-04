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

        self.num_train_batches = len(self.train_batches)
        self.num_val_batches = len(self.val_batches)
        self.num_batches = self.num_train_batches + self.num_val_batches

    def get_sample(self, index):
        x = np.zeros(self.sample_shape, 'uint8')
        for i in range(self.frames_per_sample):
            x[i] = self.vid.get_data(index + i)
        return x

    def get_samples(self, indices):
        x_shape = (len(indices),) + self.sample_shape
        x = np.zeros(x_shape, 'uint8')
        for i, index in enumerate(indices):
            x[i] = self.get_sample(index)
        return x

    def get_batch(self, is_training, batch_index):
        if is_training:
            indices = self.train_batches[batch_index]
        else:
            indices = self.val_batches[batch_index]
        return self.get_samples(indices)

    def each_batch(self):
        n = self.num_train_batches
        train_pairs = zip([1] * n, np.arange(n))
        n = self.num_val_batches
        val_pairs = zip([0] * n, np.arange(n))
        pairs = list(train_pairs) + list(val_pairs)
        np.random.shuffle(pairs)
        for is_training, batch_index in pairs:
            yield self.get_batch(is_training, batch_index)


loader = Loader('data/train.mp4')
for x in loader.each_batch():
    print(x.shape, x.dtype, x.mean(), x.min(), x.max())
