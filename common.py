import numpy as np


class Split(object):
    def __init__(self, samples_per_epoch, x_sample_shapes, x_dtypes,
                 y_sample_shapes, y_dtypes):
        self.samples_per_epoch = samples_per_epoch

        self.sample_shapes = x_sample_shapes, y_sample_shapes
        self.x_sample_shapes = x_sample_shapes
        self.y_sample_shapes = y_sample_shapes

        self.dtypes = x_dtypes, y_dtypes
        self.x_dtypes = x_dtypes
        self.y_dtypes = y_dtypes

    def batches_per_epoch(self, batch_size):
        return self.samples_per_epoch // batch_size

    def x_batch_shapes(self, batch_size):
        return [(batch_size,) + x for x in self.x_sample_shapes]

    def y_batch_shapes(self, batch_size):
        return [(batch_size,) + y for y in self.y_sample_shapes]

    def batch_shapes(self, batch_size):
        x = self.x_batch_shapes(batch_size),
        y = self.y_batch_shapes(batch_size)
        return x, y

    def get_batch(self, batch_size, index):
        raise NotImplementedError

    def shuffle(self, batch_size):
        batches_per_epoch = self.batches_per_epoch(batch_size)
        x = np.arange(batches_per_epoch)
        np.random.shuffle(x)
        return x


class RamSplit(Split):
    @classmethod
    def normalize(cls, xx):
        if isinstance(xx, np.ndarray):
            xx = [xx]
        else:
            assert isinstance(xx, (list, tuple))
        return xx

    @classmethod
    def check(cls, xx, yy):
        counts = set()
        for x in xx:
            assert isinstance(x, np.ndarray)
            counts.add(len(x))
        for y in yy:
            assert isinstance(y, np.ndarray)
            counts.add(len(y))
        assert len(counts) == 1
        assert counts.pop()

    def __init__(self, xx, yy):
        xx = self.normalize(xx)
        yy = self.normalize(yy)
        self.check(xx, yy)
        samples_per_epoch = len(xx[0])
        x_sample_shapes = [x[0].shape for x in xx]
        x_dtypes = [x[0].dtype.name for x in xx]
        y_sample_shapes = [y[0].shape for y in yy]
        y_dtypes = [y[0].dtype.name for y in yy]
        Split.__init__(self, samples_per_epoch, x_sample_shapes, x_dtypes,
                       y_sample_shapes, y_dtypes)
        self.xx = xx
        self.yy = yy

    def get_batch(self, batch_size, index):
        a = index * batch_size
        z = (index + 1) * batch_size
        batch_xx = [x[a:z] for x in self.xx]
        batch_yy = [y[a:z] for y in self.yy]
        return batch_xx, batch_yy


class Dataset(object):
    def __init__(self, train, test):
        assert isinstance(train, Split)
        if test is not None:
            assert isinstance(test, Split)
            assert train.sample_shapes == test.sample_shapes
            assert train.dtypes == test.dtypes
        self.train = train
        self.test = test

        if test:
            self.samples_per_epoch = \
                train.samples_per_epoch + test.samples_per_epoch
        else:
            self.samples_per_epoch = train.samples_per_epoch

        self.sample_shapes = train.sample_shapes
        self.x_sample_shapes = train.x_sample_shapes
        self.y_sample_shapes = train.y_sample_shapes

        self.dtypes = train.dtypes
        self.x_dtypes = train.x_dtypes
        self.y_dtypes = train.y_dtypes

    def batches_per_epoch(self, batch_size):
        batches_per_epoch = self.train.batches_per_epoch(batch_size)
        if self.test:
            batches_per_epoch += self.test.batches_per_epoch(batch_size)
        return batches_per_epoch

    def get_batch(self, batch_size, is_training, index):
        if is_training:
            split = self.train
        else:
            split = self.test
        return split.get_batch(batch_size, index)

    def shuffle(self, batch_size):
        num_train_batches = self.train.batches_per_epoch(batch_size)
        if self.test:
            num_test_batches = self.test.batches_per_epoch(batch_size)
        else:
            num_test_batches = 0
        train_batches = np.arange(num_train_batches)
        test_batches = np.arange(num_test_batches)
        x = np.zeros((num_train_batches + num_test_batches, 2), 'int64')
        x[train_batches, 0] = 1
        x[train_batches, 1] = train_batches
        x[num_train_batches + test_batches, 1] = test_batches
        np.random.shuffle(x)
        return x

    def each_batch(self, batch_size):
        for is_training, index in self.shuffle(batch_size):
            xx, yy = self.get_batch(batch_size, is_training, index)
            yield is_training, xx, yy
