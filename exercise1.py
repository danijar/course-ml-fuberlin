import numpy as np
from dataset import Mnist
from knn import KNN


def task1(dataset):
    dataset.demo()


def task2(dataset):
    for k in [1, 3, 11]:
        knn = KNN(dataset.training[:100], k)
        predictions = np.array(list(knn(x.data) for x in dataset.testing))
        error = sum(x != np.argmax(y.target) for x, y in zip(predictions,
            dataset.testing)) / len(dataset.testing)
        print('k =', k, round(100 - 100 * error, 2), '% correct')


def task3(dataset):
    pass


if __name__ == '__main__':
    dataset = Mnist()
    task1(dataset)
    task2(dataset)
    task3(dataset)
