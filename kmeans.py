import numpy as np


class KMeans:

    def __init__(self, dimensions, clusters=5):
        self.dimensions = dimensions
        self.centers = np.random.rand(clusters, dimensions)
        self.labels = np.empty(clusters)

    def train(self, examples, max_steps=20):
        for _ in range(max_steps):
            assignment = [self._assign(x) for x in examples]
            previous = self.centers.copy()
            self._center_clusters(examples, assignment)
            if (previous == self.centers).all():
                break

    def __call__(self, example):
        cluster = self._assign(example)
        return self.labels[cluster]

    def _assign(self, example):
        distances = [self._distance(example.data, x) for x in self.centers]
        return np.argmin(distances)

    def _center_clusters(self, examples, assignment):
        buckets = [[] for _ in self.centers]
        for example, cluster in zip(examples, assignment):
            buckets[cluster].append(example)
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            positions = list(map(lambda x: x.data, bucket))
            targets = list(map(lambda x: np.argmax(x.target), bucket))
            self.centers[index] = np.sum(positions, axis=0) / len(bucket)
            self.labels[index] = np.argmax(np.bincount(targets))

    def _distance(self, left, right):
        return np.linalg.norm(left - right)
