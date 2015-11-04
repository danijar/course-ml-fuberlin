import numpy as np


class KNN:

    def __init__(self, training, k=10):
        self.training = training
        self.k = k

    def __call__(self, data):
        distances = np.array(list(self._distance(data, x.data) for x in
            self.training))
        neighbors = distances.argsort()[:self.k]
        votes = np.array(list(self.training[i].target for i in neighbors))
        votes = np.argmax(votes, axis=1)
        highest = np.argmax(np.bincount(votes))
        return highest

    def _distance(self, left, right):
        return np.linalg.norm(left - right)
