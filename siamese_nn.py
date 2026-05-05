'''
Main code for training a Siamese neural network for face recognition
'''
import utils
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D
import tensorflow as tf

# ** YOUR CODE HERE **
def shared_network():
    model = Sequential()
    model.add(Input(shape=(112, 92, 1)))
    #32 filters, 3x3 kernel, relu activation, input = image
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #32 filters, 3x3 kernel, relu activation
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    return model

def siamese_nn():
    input_top = Input(shape=(112, 92, 1))
    input_bottom = Input(shape=(112, 92, 1))

    shared = shared_network()

    embedding_top = shared(input_top)
    embedding_bottom = shared(input_bottom)

    distance = Lambda(utils.euclidean_distance)(
        [embedding_top, embedding_bottom]
    )

    model = Model(
        inputs=[input_top, input_bottom],
        outputs=distance
    )

    return model


X_train, y_train, X_test, y_test = utils.get_data("att_faces")
y_test = y_test - 35
train_pairs, train_labels = utils.create_pairs(X_train, y_train, 35)
test_pairs, test_labels = utils.create_pairs(X_test, y_test, 5)
X1_train = train_pairs[:, 0]
X2_train = train_pairs[:, 1]

X1_test = test_pairs[:, 0]
X2_test = test_pairs[:, 1]

model = siamese_nn()

model.compile(
    optimizer="adam",
    loss=utils.contrastive_loss,
    metrics=[utils.accuracy]
)

model.fit(
    [X1_train, X2_train],
    train_labels,
    batch_size=32,
    epochs=10,
    validation_data=([X1_test, X2_test], test_labels)
)

model.evaluate([X1_test, X2_test], test_labels)

preds = model.predict([X1_test, X2_test])


