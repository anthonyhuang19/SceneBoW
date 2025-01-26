# Part 2:
# Nearest neighbor classifier: The nearest neighbor classifier is equally simple to
# understand. When tasked with classifying a test feature into a particular category,
# one simply finds the "nearest" training example (L2 distance is a sufficient metric)
# and assigns the test case the label of that nearest training example. The nearest
# neighbor classifier has many desirable features: it requires no training, it can
# learn arbitrarily complex decision boundaries, and it trivially supports multiclass
# problems. It is quite vulnerable to training noise, though, which can be alleviated
# by voting based on the K nearest neighbors (but you are not required to do so).
# Nearest neighbor classifiers also suffer as the feature dimensionality increases,
# because the classifier has no mechanism to learn which dimensions are irrelevant
# for the decision.
# Together, the tiny image representation and nearest neighbor classifier will get
# about 15% to 25% accuracy on the 15 scene database.
from load_image import *
import numpy as np
import scipy.spatial.distance as distance


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_clasification(train_image, train_label, test_image, k=1):
    predictions = []
    for test_sample in test_image:
        distances = [
            euclidean_distance(test_sample, train_sample)
            for train_sample in train_image
        ]
        # Getting index of the nearest k neighbors
        sorted_indices = np.argsort(distances)[:k]
        # Collecting the labels of the k nearest neighbors
        nearest_labels = [train_label[i] for i in sorted_indices]
        if k == 1:
            predictions.append(nearest_labels[0])
        else:
            predictions.append(max(set(nearest_labels), key=nearest_labels.count))
    return predictions
