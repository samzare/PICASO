import numpy as np
import h5py
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt


class ModelFetcher(object):
    def __init__(self, fname_train, fname_test, batch_size, down_sample=10):

        self.batch_size = batch_size
        self.down_sample = down_sample

        with h5py.File(fname_train, 'r') as f:
            self._train_data = np.array(f['X_train'])
            print('number of training data:', len(self._train_data))
            self._train_label = np.array(f['Y_train'])
            print('number of classes:', max(self._train_label)+1)
        with h5py.File(fname_test, 'r') as f:
            self._test_data = np.array(f['X_test'])
            print('number of test data:', len(self._test_data))
            self._test_label = np.array(f['Y_test'])

        self.num_classes = np.max(self._train_label) + 1

        self.num_train_batches = len(self._train_data) // self.batch_size
        self.num_test_batches = len(self._test_data) // self.batch_size

        self.traindata = np.zeros((self._train_data.shape[0], 64, 4, 4))
        self.testdata = np.zeros((self._test_data.shape[0], 64, 4, 4))


        for index in range(len(self._train_data)):
            img = Image.fromarray(self._train_data[index], mode='L')
            img = transforms.Resize(48)(img)
            img = transforms.CenterCrop(32)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485], [0.229])(img)

            num = 0
            for i in range(8):
                for j in range(8):
                    self.traindata[index, num] = img[0, 4 * i:4 * i + 4, 4 * j:4 * j + 4]
                    num = num + 1


        self._train_data = []
        self._train_data = self.traindata

        for index in range(len(self._test_data)):
            img = Image.fromarray(self._test_data[index], mode='L')
            img = transforms.Resize(48)(img)
            img = transforms.CenterCrop(32)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485], [0.229])(img)
            num = 0
            for i in range(8):
                for j in range(8):
                    self.testdata[index, num] = img[0, 4 * i:4 * i + 4, 4 * j:4 * j + 4]
                    num = num+1

        self._test_data = []
        self._test_data = self.testdata

        self.prep1 = lambda x: x #normalize(x)



        self._train_data = np.reshape(self._train_data, (len(self._train_data), -1, 16))
        self._test_data = np.reshape(self._test_data, (len(self._test_data), -1, 16))

        # select the subset of points to use throughout beforehand
        self.perm = np.random.permutation(self._train_data.shape[1])[::self.down_sample]

        assert len(self._train_data) > self.batch_size, \
            'Batch size larger than number of training examples'


    def train_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_data)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_label)
        return self.next_train_batch()

    def next_train_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._train_data)
        perm = self.perm
        batch_card = len(perm) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield self.prep1(self._train_data[start:end]), batch_card, self._train_label[start:end]
            start = end
            end += self.batch_size

    def test_data(self):
        return self.next_test_batch()

    def next_test_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._test_data)
        batch_card = (self._test_data.shape[1] // self.down_sample) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield self.prep1(self._test_data[start:end]), batch_card, self._test_label[start:end]
            start = end
            end += self.batch_size
