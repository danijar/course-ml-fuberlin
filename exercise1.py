import numpy as np
from dataset import Mnist
from knn import KNN
from kmeans import KMeans


def print_error(predictions, testing):
    error = sum(x != np.argmax(y.target) for x, y in zip(predictions,
        testing)) / len(testing)
    print(round(100 - 100 * error, 2), '% correct', sep='')

def task1(dataset):
    dataset.demo()


def task2(dataset):
    print('k-nn')
    for k in [1, 3, 11]:
        knn = KNN(dataset.training[:100], k)
        predictions = np.array(list(knn(x.data) for x in dataset.testing))
        print('k=', k, ' ', sep='', end='')
        print_error(predictions, dataset.testing)


def task3(dataset):
    dimensions = len(dataset.training[0].data)
    print('k-means')
    for k in [9, 10, 20]:
        kmeans = KMeans(dimensions, k)
        kmeans.train(dataset.training)
        predictions = [kmeans(x) for x in dataset.testing]
        print('k=', k, ' ', sep='', end='')
        print_error(predictions, dataset.testing)


if __name__ == '__main__':
    dataset = Mnist()
    task1(dataset)
    task2(dataset)
    task3(dataset)
