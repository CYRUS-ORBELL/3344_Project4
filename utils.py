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
    # ** YOUR CODE HERE **
    root_dir = dir
    training = []
    training_labels = []
    test = []
    test_labels = []
    #make training
    for i in range(1,35):
        folder = os.path.join(root_dir,f"s{i}")
        images = [f for f in os.listdir(folder)]
        for img_name in images:
            path = os.path.join(folder, img_name)
            img = load_img(path, color_mode="grayscale")
            img_array = img_to_array(img).squeeze()  # (H, W)
            training.append(img_array)
            training_labels.append(i)
    for i in range(36,40):
        folder = os.path.join(root_dir,f"s{i}")
        images = [f for f in os.listdir(folder)]
        for img_name in images:
            path = os.path.join(folder, img_name)
            img = load_img(path, color_mode="grayscale")
            img_array = img_to_array(img).squeeze()  # (H, W)
            test.append(img_array)
            test_labels.append(i)
    
    training = np.array(training)
    training_labels = np.array(training_labels)
    test = np.array(test)
    test_labels = np.array(test_labels)
    
    return training, training_labels, test, test_labels




def create_pairs(X,Y, num_classes):
    pairs = [] 
    labels = []
    # we are alternating same and not same images so this will tell us what to make
    same_person = True
    for i in range(num_classes):
        image1 = X[i]
        label1 = Y[i]
        if same_person:
            image2 = X[i]

            
        else:
            
            
        pairs.append((image1,image2))
        labels.append(int(same_person))
        same_person = not same_person
            

        



    return pairs, labels