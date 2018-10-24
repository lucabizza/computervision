# -*- coding: utf-8 -*-
"""
Script to index the given images
Created on 24/10/2018

@author: lucab
"""

import imutils
import cv2
import argparse
import pickle

from imutils.paths import list_images


class RGBHistogram:
    def __init__(self, bins):
        # store the number of bins the histogram will use
        self.bins = bins

    def describe(self, image):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2],
                            None, self.bins, [0, 256, 0, 256, 0, 256])
        # normalize with OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist)

        # otherwise normalize with OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist)

        # return out 3D histogram as a flattened array
        return hist.flatten()


def get_index_images():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path to the directory that contains the images to be indexed")
    ap.add_argument("-i", "--index", required=True,
                    help="Path to where the computed index will be stored")
    args = vars(ap.parse_args())

    # initialize the index dictionary to store our our quantifed
    # images, with the 'key' of the dictionary being the image
    # filename and the 'value' our computed features
    index = {}

    # initialize our image descriptor -- a 3D RGB histogram with
    # 8 bins per channel
    desc = RGBHistogram([8, 8, 8])

    # use list_images to grab the image paths and loop over them
    for imagePath in list_images(args["dataset"]):
        # extract our unique image ID (i.e. the filename)
        k = imagePath[imagePath.rfind("/") + 1:]

        # load the image, describe it using our RGB histogram
        # descriptor, and update the index
        image = cv2.imread(imagePath)
        features = desc.describe(image)
        index[k] = features

    # we are now done indexing our image -- now we can write our
    # index to disk
    f = open(args["index"], "wb")
    f.write(pickle.dumps(index))
    f.close()

    # show how many images we indexed
    print("[INFO] done...indexed {} images".format(len(index)))


if __name__ == "__main__":
    get_index_images()
