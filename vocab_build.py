# Part 3:
# Building the vocabulary of visual words: Before we can represent our training and
# testing images as bag of feature histograms, we first need to establish a vocabulary of
# visual words. We will form this vocabulary by sampling many local features from our
# training set (10's or 100's of thousands) and then clustering them with kmeans. The
# number of kmeans clusters is the size of our vocabulary and the size of our features. For example, you might start by clustering many SIFT descriptors into k=50 clusters. This partitions the continuous, 128 dimensional SIFT feature space into 50 regions. For any new SIFT feature we observe, we can figure out which region it belongs to as
# long as we save the centroids of our original clusters. Those centroids are our visual
# word vocabulary. You can use any existing libraries for kmeans.

import cv2
import numpy as np
from sklearn.cluster import KMeans
import joblib


def vocab_build(image_paths, vocab_size, save_model_path="kmeans_model.pkl"):
    sift = cv2.SIFT_create()
    bag_of_features = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            bag_of_features.append(descriptors)

    bag_of_features = np.vstack(bag_of_features)

    print("Making a new model kmean")
    kmeans = KMeans(n_clusters=vocab_size, random_state=0)
    kmeans.fit(bag_of_features)
    print(f"Saved k-means model to {save_model_path}")
    return kmeans
