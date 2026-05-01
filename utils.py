'''
Helper functions for face recognition
'''
import numpy as np
import random
import os
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array

def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(Y_true, D):
    margin = 1
    return K.mean(Y_true * K.square(D) + (1 - Y_true) * K.maximum((margin-D),0))

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def get_data(dir):
    X_train, y_train = [], []
    X_test, y_test = [], []

    # loop through s1 → s40
    for i in range(1, 41):
        folder = os.path.join(dir, f"s{i}")
        images = sorted([f for f in os.listdir(folder) if f.endswith(".pgm")])

        for img_name in images:
            path = os.path.join(folder, img_name)

            # load image
            img = load_img(path, color_mode="grayscale")
            img_array = img_to_array(img).squeeze()  # (H, W)

            # assign to train or test
            if i <= 35:
                X_train.append(img_array)
                y_train.append(i)
            else:
                X_test.append(img_array)
                y_test.append(i)

    # convert to numpy arrays (important for ML)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

    

def create_pairs(X,Y, num_classes):
    # ** YOUR CODE HERE **
    