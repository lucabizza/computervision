# -*- coding: utf-8 -*-
"""
Step by step guide to ResNet Neural Network

Created on 13/11/2018

@author: lucab
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(5678)
np.set_printoptions(precision=2, suppress=True)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


def ReLu(x):
    mask = (x > 0) * 1.0
    return mask * x


def d_ReLu(x):
    mask = (x > 0) * 1.0
    return mask


def log(x):
    return 1 / (1 + np.exp(-1 * x))


def d_log(x):
    return log(x) * (1 - log(x))


def arctan(x):
    return np.arctan(x)


def d_arctan(x):
    return 1 / (1 + x ** 2)


def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp / exp.sum()


# 1. Read Data
mnist = input_data.read_data_sets("./mnist_data/", one_hot=True).test
images, label = shuffle(mnist.images, mnist.labels)
test_image_num, training_image_num = 50, 800
learning_rate = 0.001
learning_rate_h = 0.00001
num_epoch = 100
testing_images, testing_lables = images[:test_image_num, :], label[:test_image_num, :]
training_images, training_lables = images[test_image_num:test_image_num + training_image_num, :], \
                                   label[test_image_num: test_image_num + training_image_num, :]

# 2. Hyper Parameters
w1a = np.random.randn(784, 1024)
w1b = np.random.randn(1024, 824)
w1h = np.random.randn(784, 824)

w2a = np.random.randn(824, 512)
w2b = np.random.randn(512, 256)
w2h = np.random.randn(824, 256)

w3a = np.random.randn(256, 128)
w3b = np.random.randn(128, 10) * 0.01
w3h = np.random.randn(256, 10) * 0.01

cost_array = []
total_cost = 0
mid_correct = 0

for iter in range(num_epoch):

    for current_image_index in range(len(training_images)):
        current_image = np.expand_dims(training_images[current_image_index, :], axis=0)
        current_label = np.expand_dims(training_lables[current_image_index, :], axis=0)

        #         # ------ Res Block 1 ------
        Rl1a = current_image.dot(w1a)
        Rl1aA = arctan(Rl1a)
        Rl1b = Rl1aA.dot(w1b)

        RH1 = current_image.dot(w1h)
        RH1A = tanh(RH1)

        H1 = Rl1b + RH1A
        H1A = ReLu(H1)
        #

        #         # ------ Res Block 2 ------
        R21a = H1A.dot(w2a)
        R21aA = arctan(R21a)
        R21b = R21aA.dot(w2b)

        RH2 = H1A.dot(w2h)
        RH2A = tanh(RH2)

        H2 = R21b + RH2A
        H2A = ReLu(H2)
        #

        #         # ------ Res Block 3 ------
        R31a = H2A.dot(w3a)
        R31aA = arctan(R31a)
        R31b = R31aA.dot(w3b)

        RH3 = H2A.dot(w3h)
        RH3A = tanh(RH3)

        H3 = R31b + RH3A
        H3A = ReLu(H3)
        #

        # ---- Cost -----
        H3Soft = softmax(H3A)
        cost = (-(current_label * np.log(H3Soft) + (1 - current_label) * np.log(1 - H3Soft))).sum()
        print("Real Time Update Cost: ", cost, end='\r')
        total_cost += cost
        # ---- Cost -----

        # ---- Grad 3 -----
        grad_3_part_common = (H3Soft - current_label) * d_ReLu(H3)

        grad_3H_part_2 = d_tanh(RH3)
        grad_3H_part_3 = H2A
        grad_3H = grad_3H_part_3.T.dot(grad_3_part_common * grad_3H_part_2)

        grad_3w2_part_3 = R31aA
        grad_3w2 = grad_3w2_part_3.T.dot(grad_3_part_common)

        grad_3w1_part_1 = (grad_3_part_common).dot(w3b.T)
        grad_3w1_part_2 = d_arctan(R31a)
        grad_3w1_part_3 = H2A
        grad_3w1 = grad_3w1_part_3.T.dot(grad_3w1_part_1 * grad_3w1_part_2)
        # ---- Grad 3 -----

        # ---- Grad 2 -----
        grad_2_part_common = ((grad_3w1_part_1 * grad_3w1_part_2).dot(w3a.T) + (
                grad_3_part_common * grad_3H_part_2).dot(w3h.T)) * d_ReLu(H2)

        grad_2H_part_2 = d_tanh(RH2)
        grad_2H_part_3 = H1A
        grad_2H = grad_2H_part_3.T.dot(grad_2_part_common * grad_2H_part_2)

        grad_2w2_part_3 = R21aA
        grad_2w2 = grad_2w2_part_3.T.dot(grad_2_part_common)

        grad_2w1_part_1 = (grad_2_part_common).dot(w2b.T)
        grad_2w1_part_2 = d_arctan(R21a)
        grad_2w1_part_3 = H1A
        grad_2w1 = grad_2w1_part_3.T.dot(grad_2w1_part_1 * grad_2w1_part_2)
        # ---- Grad 2 -----

        # ---- Grad 1 -----
        grad_1_part_common = ((grad_2w1_part_1 * grad_2w1_part_2).dot(w2a.T) + (
                grad_2_part_common * grad_2H_part_2).dot(w2h.T)) * d_ReLu(H1)

        grad_1H_part_2 = d_tanh(RH1)
        grad_1H_part_3 = current_image
        grad_1H = grad_1H_part_3.T.dot(grad_1_part_common * grad_1H_part_2)

        grad_1w2_part_3 = Rl1aA
        grad_1w2 = grad_1w2_part_3.T.dot(grad_1_part_common)

        grad_1w1_part_1 = (grad_1_part_common).dot(w1b.T)
        grad_1w1_part_2 = d_arctan(Rl1a)
        grad_1w1_part_3 = current_image
        grad_1w1 = grad_1w1_part_3.T.dot(grad_1w1_part_1 * grad_1w1_part_2)
        # ---- Grad 1 -----

        w3h = w3h - learning_rate_h * grad_3H
        w3b = w3b - learning_rate * grad_3w2
        w3a = w3a - learning_rate * grad_3w1

        w2h = w2h - learning_rate_h * grad_2H
        w2b = w2b - learning_rate * grad_2w2
        w2a = w2a - learning_rate * grad_2w1

        w1h = w1h - learning_rate_h * grad_1H
        w1b = w1b - learning_rate * grad_1w2
        w1a = w1a - learning_rate * grad_1w1

    if iter % 2 == 0:
        print("current Iter: ", iter, " Current Cost :", total_cost / len(training_images))

        for current_batch_index in range(30):

            testing_images, testing_lables = shuffle(testing_images, testing_lables)

            current_batch = np.expand_dims(testing_images[current_batch_index, :], axis=0)
            current_batch_label = testing_lables[current_batch_index, :]

            #
            Rl1a = current_batch.dot(w1a)
            Rl1aA = arctan(Rl1a)
            Rl1b = Rl1aA.dot(w1b)

            RH1 = current_image.dot(w1h)
            RH1A = tanh(RH1)

            H1 = Rl1b + RH1A
            H1A = ReLu(H1)
            #

            #
            R21a = H1A.dot(w2a)
            R21aA = arctan(R21a)
            R21b = R21aA.dot(w2b)

            RH2 = H1A.dot(w2h)
            RH2A = tanh(RH2)

            H2 = R21b + RH2A
            H2A = ReLu(H2)
            #

            #
            R31a = H2A.dot(w3a)
            R31aA = arctan(R31a)
            R31b = R31aA.dot(w3b)

            RH3 = H2A.dot(w3h)
            RH3A = tanh(RH3)

            H3 = R31b + RH3A
            H3A = ReLu(H3)
            #

            H3Soft = softmax(H3A)
            try:
                if np.where(H3Soft[0] == H3Soft[0].max())[0] == \
                        np.where(current_batch_label == current_batch_label.max())[0]:
                    mid_correct += 1
            except:
                pass
            print('Current Predict : ',
                  np.where(H3Soft[0] == H3Soft[0].max())[0],
                  "Ground Truth   : ", np.where(current_batch_label == current_batch_label.max())[0]
                  # '\n',
                  # l7Soft,'\n',current_batch_label
                  )

        print("Correct Classitication: ", mid_correct, " / 30")
        mid_correct = 0
        print('------------------------')
    cost_array.append(total_cost / len(training_images))
    total_cost = 0

print('=======================================')
print('==============FINAL====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_batch = np.expand_dims(testing_images[current_batch_index, :], axis=0)
    current_batch_label = testing_lables[current_batch_index, :]

    Rl1a = current_batch.dot(w1a)
    Rl1aA = arctan(Rl1a)
    Rl1b = Rl1aA.dot(w1b)

    RH1 = current_image.dot(w1h)
    RH1A = tanh(RH1)

    H1 = Rl1b + RH1A
    H1A = ReLu(H1)
    #

    #
    R21a = H1A.dot(w2a)
    R21aA = arctan(R21a)
    R21b = R21aA.dot(w2b)

    RH2 = H1A.dot(w2h)
    RH2A = tanh(RH2)

    H2 = R21b + RH2A
    H2A = ReLu(H2)
    #

    #
    R31a = H2A.dot(w3a)
    R31aA = arctan(R31a)
    R31b = R31aA.dot(w3b)

    RH3 = H2A.dot(w3h)
    RH3A = tanh(RH3)

    H3 = R31b + RH3A
    H3A = ReLu(H3)
    #

    H3Soft = softmax(H3A)

    print(' Current Predict : ', np.where(H3Soft[0] == H3Soft[0].max())[0], " Ground Truth : ",
          np.where(current_batch_label == current_batch_label.max())[0])

    if np.where(H3Soft[0] == H3Soft[0].max())[0] == np.where(current_batch_label == current_batch_label.max())[0]:
        correct += 1

print('Correct : ', correct, ' Out of : ', len(testing_images))
plt.title("Cost over time")
plt.plot(np.arange(num_epoch), cost_array)
plt.show()
