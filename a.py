import imageio
import numpy as np


class Loader(object):
    def __init__(self, filename, frames_per_sample=8, val_frac=0.2):
        self.filename = filename
        self.frames_per_sample = frames_per_sample
        self.vid = imageio.get_reader(filename, 'ffmpeg')
        self.num_samples = len(self.vid) - frames_per_sample + 1
        im = self.vid.get_data(0)
        self.sample_shape = (self.frames_per_sample,) + im.shape
        self.is_sample_train = \
            val_frac < np.random.uniform(0, 1, self.num_samples)

    def get_sample(self, index):
        x = np.zeros(self.sample_shape, 'uint8')
        for i in range(self.frames_per_sample):
            x[i] = self.vid.get_data(index + i)
        return x


loader = Loader('data/train.mp4')
x = loader.get_sample(1337)
print(x.shape, x.min(), x.max(), x.mean(), x.dtype)
