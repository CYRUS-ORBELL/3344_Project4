'''
Helper functions for face recognition
'''
import numpy as np
import random
import os
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
#new stuff
import tensorflow as tf



def euclidean_distance(vectors):
    v1, v2 = vectors
    return tf.sqrt(
        tf.reduce_sum(tf.square(v1 - v2), axis=1) 
    )

def contrastive_loss(y_true, d):
    y_true = tf.cast(y_true, tf.float32)
    margin = 1.0  
    return tf.reduce_mean(
        y_true * tf.square(d) +
        (1 - y_true) * tf.square(tf.maximum(margin - d, 0.0))
    )

def accuracy(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1])  # flatten to 1D
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, tf.float32)), tf.float32))


def get_data(dir):
    X_train, y_train = [], []
    X_test, y_test = [], []
    for subject in range(1, 41):
        folder = os.path.join(dir, f"s{subject}")
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            # load image in grayscale
            img = Image.open(img_path).convert("L")
            img = np.array(img)

            # normalize (important for neural nets)
            img = img / 255.0

            # add channel dimension (112x92 -> 112x92x1)
            img = np.expand_dims(img, axis=-1)

            if subject <= 35:
                X_train.append(img)
                y_train.append(subject)
            else:
                X_test.append(img)
                y_test.append(subject)
    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
    )

def create_pairs(X, Y, num_classes, return_subject_ids=False):
    pairs = []
    labels = []
    subject_ids = []
    X = np.array(X)
    Y = np.array(Y)
    
    class_indices = [np.where(Y == i)[0] for i in range(1, num_classes + 1)]

    for c in range(num_classes):
        idxs = class_indices[c]
        # All same-class combinations instead of just sequential
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pairs.append([X[idxs[i]], X[idxs[j]]])
                labels.append(1)
                subject_ids.append((Y[idxs[i]], Y[idxs[j]]))
                # One different-class pair per same-class pair
                c2 = (c + np.random.randint(1, num_classes)) % num_classes
                idxs2 = class_indices[c2]
                idx2 = idxs2[np.random.randint(0, len(idxs2))]
                pairs.append([X[idxs[i]], X[idx2]])
                labels.append(0)
                subject_ids.append((Y[idxs[i]], Y[idx2]))

    if return_subject_ids:
        return np.array(pairs), np.array(labels), np.array(subject_ids)
    return np.array(pairs), np.array(labels)

