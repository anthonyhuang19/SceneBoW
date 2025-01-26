# Part 4:
# Building the vocabulary of visual words: Once the vocabulary is built, we are ready
# to represent our training and testing images as histograms of visual words. For each
# image we will densely sample many SIFT descriptors. Instead of storing hundreds of
# SIFT descriptors, we simply count how many SIFT descriptors fall into each cluster in
# our visual word vocabulary. This is done by finding the nearest neighbor kmeans
# centroid for every SIFT feature. Thus, if we have a vocabulary of 50 visual words, and we detect 220 SIFT features in an image, our bag of SIFT representation will be a
# histogram of 50 dimensions where each bin counts how many times a SIFT descriptor
# was assigned to that cluster and sums to 220. The histogram should be normalized so
# that image size does not dramatically change the bag of feature magnitude.
import cv2
import numpy as np


def create_bag_of_sifts(image_paths, kmeans):
    f_hist = []
    for images in image_paths:
        #using SIFT algorihtm
        image = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            visual_words = kmeans.predict(descriptors)
            # Create histogram of visual word occurrences
            hist, _ = np.histogram(
                visual_words, bins=np.arange(kmeans.n_clusters + 1), density=True
            )
            # normalize histogram
            hist_norm = [float(i) / sum(hist) for i in hist]
            f_hist.append(hist_norm)
    image_feats = np.asarray(f_hist)
    return image_feats
