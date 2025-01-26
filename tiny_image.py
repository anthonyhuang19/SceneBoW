# Part 1
# "Tiny image" feature: The "tiny image" feature is one of the simplest possible
# image representations. One simply resizes each image to a small, fixed resolution
# (we recommend 16x16). It works slightly better if the tiny image is made to have
# zero mean and unit length. This is not a particularly good representation, because
# it discards all of the high frequency image content and is not especially shift
# invariant.

from PIL import Image
import numpy as np
from load_image import *


def tiny_image(image_path):
    # Getting the size of image in path
    n = len(image_path)
    # the size of pixel in image  for tiny image
    size = 16

    tiny_images = []

    for image in image_path:
        picture = Image.open(image)
        picture = picture.resize((size, size))
        picture = (picture - np.mean(picture)) / np.std(picture)
        picture = picture.flatten()
        tiny_images.append(picture)
    return np.asarray(tiny_images)


# if __name__ == '__main__':
#     train,test = load_dataset()
#     output = tiny_image(train)
#     print(output[0].shape)
