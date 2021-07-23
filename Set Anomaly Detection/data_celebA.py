"""adapted from:
https://github.com/juho-lee/set_transformer
"""


import numpy as np
import h5py
from PIL import Image
from torchvision import transforms
import torch
import os


def normalize(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)   #x / 255.  #


class ModelFetcher(object):
    def __init__(self, fname_train, fname_test, batch_size, down_sample=10):

        # self.fname = fname
        self.batch_size = batch_size
        self.down_sample = down_sample

        self._train_data = np.load(fname_train)['data']
        self._train_label = np.load(fname_train)['label']

        self._test_data = np.load(fname_test)['data']
        self._test_label = np.load(fname_test)['label']

        self.traindata = np.zeros((self._train_data.shape[0], 8, 3, 64, 64))
        self.trainlabel = np.zeros(self._train_label.shape[0])
        self.testdata = np.zeros((self._test_data.shape[0], 8, 3, 64, 64))
        self.testlabel = np.zeros(self._test_label.shape[0])

        for index in range(len(self._train_data)):
            rng_state = np.random.get_state()
            np.random.shuffle(self._train_data[index])
            np.random.set_state(rng_state)
            np.random.shuffle(self._train_label[index])
            for j in range(8):
                #if self._train_label[index][j] == 1:
                    #self.trainlabel[index] = j

                img = Image.open(os.path.join('./celebA/Img/img_align_celeba_png', self._train_data[index][j]))
                img = transforms.CenterCrop(160)(img)
                img = transforms.Resize((64, 64))(img)
                img = transforms.RandomHorizontalFlip()(img)
                img = transforms.RandomRotation(60, resample=Image.BILINEAR)(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.4810, 0.4574, 0.4078], [0.2605, 0.2533, 0.2684])(img)
                self.traindata[index][j] = (img)

        self._train_data = []
        self._train_data = self.traindata
        #self._train_label = []
        #self._train_label = self.trainlabel

        for index in range(len(self._test_data)):
            rng_state1 = np.random.get_state()
            np.random.shuffle(self._test_data[index])
            np.random.set_state(rng_state1)
            np.random.shuffle(self._test_label[index])
            for j in range(8):
                #if self._test_label[index][j] == 1:
                    #self.testlabel[index] = j

                img = Image.open(os.path.join('./celebA/Img/img_align_celeba_png', self._test_data[index][j]))
                img = transforms.CenterCrop(160)(img)
                img = transforms.Resize((64, 64))(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.4810, 0.4574, 0.4078], [0.2605, 0.2533, 0.2684])(img)
                self.testdata[index][j] = (img)

        self._test_data = []
        self._test_data = self.testdata
        #self._test_label = []
        #self._test_label = self.testlabel

        self.prep1 = lambda x: x #normalize(x) #x


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
        rng_state = np.random.get_state()
        np.random.shuffle(self._test_data)
        np.random.set_state(rng_state)
        np.random.shuffle(self._test_label)
        return self.next_test_batch()

    def next_test_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._test_data)
        while end < N:
            yield self.prep1(self._test_data[start:end]), self._test_label[start:end]
            start = end
            end += self.batch_size
