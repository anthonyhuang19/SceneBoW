# SceneBoW

[![Python][python-badge]][python-url]
[![License][license-badge]][license-url]

*👀 Scene recognition using Bag of Words and Tiny Image representation*

## Overview

**SceneBoW** performs scene recognition using two classical computer vision methods:

* **Bag of Words (BoW)** with **SIFT** features
* **Tiny Image Representation**

Both methods are combined with **k-Nearest Neighbors (KNN)** on the **15 Scene Dataset**.

### Bag of Words (BoW)

* Extract **SIFT** features from images
* Cluster features into "visual words" using **k-means**
* Represent each image as a histogram of visual words
* Classify images using **KNN**

### Tiny Image Representation

* Resize images to low resolution (tiny images)
* Represent as feature vectors
* Classify using **BoW + KNN**

## Dataset: 15 Scene Categories

Includes categories such as Bedroom, Coast, Forest, Highway, Kitchen, Mountain, Office, and more.

### Sample Images

| Category | Image                                                         | Category | Image                                                              |
| -------- | ------------------------------------------------------------- | -------- | ------------------------------------------------------------------ |
| Bedroom  | ![bedroom](/result/Bag%20of%20Words/bag_words/bedroom_fn.png) | Kitchen  | ![kitchen](/result/Bag%20of%20Words/bag_words/kitchen_train.png)   |
| Coast    | ![coast](/result/Bag%20of%20Words/bag_words/coast_train.png)  | Mountain | ![mountain](/result/Bag%20of%20Words/bag_words/mountain_train.png) |

## Feature Extraction

### SIFT (Scale-Invariant Feature Transform)

* Extracts robust keypoints for BoW representation

### Tiny Images

* Resize images to fixed low resolution
* Fast and lightweight for classification

## Classification with KNN

* Uses majority vote from `k` nearest neighbors
* Applies to both BoW and Tiny Image features

### Classification Results

| Category | TP                                                     | FP                                                     | FN                                                     |
| -------- | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ |
| Bedroom  | ![](/result/Bag%20of%20Words/bag_words/bedroom_tp.png) | ![](/result/Bag%20of%20Words/bag_words/bedroom_fp.png) | ![](/result/Bag%20of%20Words/bag_words/bedroom_fn.png) |
| Kitchen  | ![](/result/Bag%20of%20Words/bag_words/kitchen_tp.png) | ![](/result/Bag%20of%20Words/bag_words/kitchen_fp.png) | ![](/result/Bag%20of%20Words/bag_words/kitchen_fn.png) |

### Tiny Image Results

| Category | TP                                                | FP                                                | FN                                                |
| -------- | ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Bedroom  | ![](/result/tiny_image/tiny_image/bedroom_tp.png) | ![](/result/tiny_image/tiny_image/bedroom_fp.png) | ![](/result/tiny_image/tiny_image/bedroom_fn.png) |
| Kitchen  | ![](/result/tiny_image/tiny_image/kitchen_tp.png) | ![](/result/tiny_image/tiny_image/kitchen_fp.png) | ![](/result/tiny_image/tiny_image/kitchen_fn.png) |

## Performance Comparison

Accuracy and computational cost comparison between BoW + KNN and Tiny Image + KNN.

![Performance](2.png)

## Main Files

| File                 | Description                            |
| -------------------- | -------------------------------------- |
| `create_bag_sift.py` | Extracts SIFT features and creates BoW |
| `knn.py`             | KNN classification logic               |
| `evaluation.py`      | Measures accuracy and performance      |
| `load_image.py`      | Loads and preprocesses images          |
| `main.py`            | Runs the full pipeline                 |

## Conclusion

**SceneBoW** demonstrates the effectiveness of traditional methods like **SIFT**, **Bag of Words**, and **Tiny Images** combined with **KNN** for interpretable scene recognition.

---

[python-badge]: https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white
[python-url]: https://www.python.org/
[license-badge]: https://img.shields.io/badge/license-MIT-green
[license-url]: https://opensource.org/licenses/MIT
