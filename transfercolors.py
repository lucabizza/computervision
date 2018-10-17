# -*- coding: utf-8 -*-
"""
Script to transfer the colors background from one image to another
Created on 17/10/2018

@author: lucab
"""
import argparse

import cv2
import numpy as np


def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array

    :param arr: NumPy array to be scaled to [new_min, new_max] range
    :param new_range: tuple of form (min, max) specifying range of
                      transformed array
    :return scaled: NumPy array that has been scaled to be in
                    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.

    :param arr: array to be trimmed to [0, 255] range
    :param clip: should array be scaled by np.clip? if False then input
                 array will be min-max scaled to range
                 [max([arr.min(), 0]), min([arr.max(), 255])]
    :return scaled: NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled


def color_transfer(source, target, clip=True):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.
    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    :param source: NumPy array, OpenCV image in BGR color space (the source image)
    :param target: NumPy array, OpenCV image in BGR color space (the target image)
    :param clip: Should components of L*a*b* image be scaled by np.clip before
                 converting back to BGR color space?
                 If False then components will be min-max scaled appropriately.
                 Clipping will keep target image brightness truer to the input.
                 Scaling will adjust image brightness to avoid washed out portions
                 in the resulting color transfer that can be caused by clipping.
    :return transfer: NumPy array OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)
    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    # return the color transferred image
    return transfer


def show_image(title, image, width=300):
    # resize the image to have a constant width, just to
    # make displaying the images take up less screen real
    # estate
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # show the resized image
    cv2.imshow(title, resized)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def transfer_color():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=True,
                    help="Path to the source image")
    ap.add_argument("-t", "--target", required=True,
                    help="Path to the target image")
    ap.add_argument("-c", "--clip", type=str2bool, default='t',
                    help="Should np.clip scale L*a*b* values before final conversion to BGR? "
                         "Appropriate min-max scaling used if False.")
    ap.add_argument("-o", "--output", help="Path to the output image (optional)")
    args = vars(ap.parse_args())

    # load the images
    source = cv2.imread(args["source"])
    target = cv2.imread(args["target"])

    # transfer the color distribution from the source image
    # to the target image
    transfer = color_transfer(source, target, clip=args["clip"])

    # check to see if the output image should be saved
    if args["output"] is not None:
        cv2.imwrite(args["output"], transfer)

    # show the images and wait for a key press
    show_image("Source", source)
    show_image("Target", target)
    show_image("Transfer", transfer)
    cv2.waitKey(0)


if __name__ == "__main__":
    transfer_color()
