# -*- coding: utf-8 -*-
"""
Created on 14/11/2018

@author: lucab
"""
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras.models import load_model
import numpy as np
import argparse
import cv2


def get_logo_detected():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to pre-trained model")
    ap.add_argument("-l", "--labels", required=True,
                    help="path to class labels")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load the class label mappings
    LABELS = open(args["labels"]).read().strip().split("\n")
    LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

    # load the model from disk
    model = load_model(args["model"], custom_objects=custom_objects)

    # load the input image (in BGR order), clone it, and preprocess it
    image = read_image_bgr(args["image"])
    output = image.copy()
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    # detect objects in the input image and correct for the image scale
    (_, _, boxes, nmsClassification) = model.predict_on_batch(image)
    boxes /= scale

    # compute the predicted labels and probabilities
    predLabels = np.argmax(nmsClassification[0, :, :], axis=1)
    scores = nmsClassification[0,
                               np.arange(0, nmsClassification.shape[1]), predLabels]

    # loop over the detections
    for (i, (label, score)) in enumerate(zip(predLabels, scores)):
        # filter out weak detections
        if score < args["confidence"]:
            continue

        # grab the bounding box for the detection
        box = boxes[0, i, :].astype("int")

        # build the label and draw the label + bounding box on the output
        # image
        label = "{}: {:.2f}".format(LABELS[label], score)
        cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), 2)
        cv2.putText(output, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    get_logo_detected()
