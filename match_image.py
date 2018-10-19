# -*- coding: utf-8 -*-
"""
Script to do matching template on an image

Created on 19/10/2018

@author: lucab
"""

import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--background", required=True,
                help="Path to the background image")
ap.add_argument("-w", "--image_to_match", required=True,
                help="Path to the image_to_match image")
args = vars(ap.parse_args())

# load the background and image_to_match images
background = cv2.imread(args["puzzle"])
image_to_match = cv2.imread(args["waldo"])
(image_to_matchHeight, image_to_matchWidth) = image_to_match.shape[:2]

# find the image_to_match in the background
result = cv2.matchTemplate(background, image_to_match, cv2.TM_CCOEFF)
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

# the background image
topLeft = maxLoc
botRight = (topLeft[0] + image_to_matchWidth, topLeft[1] + image_to_matchHeight)
roi = background[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the background except for image_to_match
mask = np.zeros(background.shape, dtype="uint8")
background = cv2.addWeighted(background, 0.25, mask, 0.75, 0)

# put the original image_to_match back in the image so that he is
# 'brighter' than the rest of the image
background[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

# display the images
cv2.imshow("Background", imutils.resize(background, height=650))
cv2.imshow("Image to match", image_to_match)
cv2.waitKey(0)
