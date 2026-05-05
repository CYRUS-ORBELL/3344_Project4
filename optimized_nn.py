'''
Main code for training a Siamese neural network for face recognition
'''
import utils
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D
import tensorflow as tf
from keras.optimizers import Adam
# ** YOUR CODE HERE **
def shared_network():
    model = Sequential()
    model.add(Input(shape=(112, 92, 1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
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
    optimizer=Adam(learning_rate=0.0003, clipnorm=1.0),
    loss=utils.contrastive_loss,
    metrics=[utils.accuracy]
)

model.fit(
    [X1_train, X2_train],
    train_labels,
    batch_size=32,
    epochs=20,  # more epochs
    validation_data=([X1_test, X2_test], test_labels)
)
preds = model.predict([X1_test, X2_test])
print("preds shape:", preds.shape)
model.evaluate([X1_test, X2_test], test_labels)
print("\n--- DEBUG ---")
print("Sample predictions:", preds[:10].flatten())
print("Sample labels:", test_labels[:10])
print("Pred < 0.5:", (preds[:10].flatten() < 0.5))
print("Pred < 1.0:", (preds[:10].flatten() < 1.0))




