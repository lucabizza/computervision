# -*- coding: utf-8 -*-
"""
Script to search from an image given an index
Created on 24/10/2018

@author: lucab
"""
import numpy as np
import argparse
import os
import pickle
import cv2


class Searcher:
    def __init__(self, index):
        # store our index of images
        self.index = index

    def search(self, queryFeatures):
        # initialize our dictionary of results
        results = {}

        # loop over the index
        for (k, features) in self.index.items():
            # compute the chi-squared distance between the features
            # in our index and our query features -- using the
            # chi-squared distance which is normally used in the
            # computer vision field to compare histograms
            d = self.chi2_distance(features, queryFeatures)

            # now that we have the distance between the two feature
            # vectors, we can udpate the results dictionary -- the
            # key is the current image ID in the index and the
            # value is the distance we just computed, representing
            # how 'similar' the image in the index is to our query
            results[k] = d

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])

        return results

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d


def get_results_search():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path to the directory that contains the images we just indexed")
    ap.add_argument("-i", "--index", required=True,
                    help="Path to where we stored our index")
    args = vars(ap.parse_args())

    # load the index and initialize our searcher
    index = pickle.loads(open(args["index"], "rb").read())
    searcher = Searcher(index)

    # loop over images in the index -- we will use each one as
    # a query image
    for (query, queryFeatures) in index.items():
        # perform the search using the current query
        results = searcher.search(queryFeatures)

        # load the query image and display it
        path = os.path.join(args["dataset"], query)
        queryImage = cv2.imread(path)
        cv2.imshow("Query", queryImage)
        print("query: {}".format(query))

        # initialize the two montages to display our results --
        # we have a total of 25 images in the index, but let's only
        # display the top 10 results; 5 images per montage, with
        # images that are 400x166 pixels
        montageA = np.zeros((166 * 5, 400, 3), dtype="uint8")
        montageB = np.zeros((166 * 5, 400, 3), dtype="uint8")

        # loop over the top ten results
        for j in range(0, 10):
            # grab the result (we are using row-major order) and
            # load the result image
            (score, imageName) = results[j]
            path = os.path.join(args["dataset"], imageName)
            result = cv2.imread(path)
            print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

            # check to see if the first montage should be used
            if j < 5:
                montageA[j * 166:(j + 1) * 166, :] = result

            # otherwise, the second montage should be used
            else:
                montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result

        # show the results
        cv2.imshow("Results 1-5", montageA)
        cv2.imshow("Results 6-10", montageB)
        cv2.waitKey(0)


if __name__ == "__main__":
    get_results_search()
