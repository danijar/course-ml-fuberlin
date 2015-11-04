import array
import gzip
import os
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from example import Example


class Dataset:

    def __init__(self):
        self.training = []
        self.testing = []

    def split(self, examples, ratio=0.8):
        split = int(ratio * len(examples))
        return examples[:split], examples[split:]


class Mnist(Dataset):

    def __init__(self, path='dataset/mnist'):
        self.training = self.read(
            os.path.join(path, 'train-images-idx3-ubyte.gz'),
            os.path.join(path, 'train-labels-idx1-ubyte.gz'))
        self.testing = self.read(
            os.path.join(path, 't10k-images-idx3-ubyte.gz'),
            os.path.join(path, 't10k-labels-idx1-ubyte.gz'))

    def read(self, image_path, label_path):
        images = gzip.open(image_path, 'rb')
        _, size, rows, cols = struct.unpack('>IIII', images.read(16))
        image_bin = array.array('B', images.read())
        images.close()

        labels = gzip.open(label_path, 'rb')
        _, size2 = struct.unpack('>II', labels.read(8))
        assert size == size2
        label_bin = array.array('B', labels.read())
        labels.close()

        examples = []
        for i in range(size):
            data = image_bin[i*rows*cols:(i+1)*rows*cols]
            data = np.array(data).reshape(rows * cols) / 255
            target = np.zeros(10)
            target[label_bin[i]] = 1
            examples.append(Example(data, target))
        return examples

    def show(self, example):
        _, ax = plt.subplot(111)
        self._plot(example, ax)
        plt.show()

    def demo(self, shape=(2, 5)):
        _, axes = plt.subplots(*shape)
        for ax in axes.flatten():
            example = random.choice(self.training)
            self._plot(example, ax)
        plt.show()

    def _plot(self, example, ax):
        ax.imshow(example.data.reshape(28, 28), cmap=cm.gray,
            interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(np.argmax(example.target))
