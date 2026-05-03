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

    #pooling the paper has only one pooling layer but recource online told me we should have two as they reduce noise
    model.add(MaxPooling2D(pool_size=(2,2)))

    #flatten
    model.add(Flatten())

    #dense layer
    model.add(Dense(128,activation= "sigmoid" ))

    return model

def siamese_model():
    input_top = Input(shape=(112, 92, 1))
    input_bottom = Input(shape=(112, 92, 1))

    shared = shared_network()
  
    embedding_top = shared(input_top)
    embedding_bottom = shared(input_bottom)
   
    distance = Lambda(utils.euclidean_distance)([embedding_top, embedding_bottom])

    model = Model(
    inputs=[input_top, input_bottom],
    outputs=distance
    )

    return model




    
