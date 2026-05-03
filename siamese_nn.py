'''
Main code for training a Siamese neural network for face recognition
'''
import utils
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D

# ** YOUR CODE HERE **



def shared_network():
    model= Sequential()
    #conv layer 1
    model.add(Conv2D(filters = 32,  kernel_size =(3,3), activation="relu", input_shape = (112, 92, 1)))

    #pooling 
    model.add(MaxPooling2D(pool_size=(2,2)))

    #conv layer 2
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation="relu"))

    #pooling 
    model.add(MaxPooling2D(pool_size=(2,2)))

    #flatten
    model.add(Flatten())

    #dense layer
    model.add(Dense(128,activation= "sigmoid" ))



    
