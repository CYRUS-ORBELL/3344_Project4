
'''
Main code for training a Siamese neural network for face recognition
'''
import utils
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D,Dropout
import tensorflow as tf
from keras.optimizers import Adam
# ** YOUR CODE HERE **

def shared_network():
    model = Sequential()

    model.add(Input(shape=(112, 92, 1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
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
    batch_size=10,
    epochs=4,  # more epochs
    validation_data=([X1_test, X2_test], test_labels)
)
model.evaluate([X1_test, X2_test], test_labels)






#------------------------------visual stuff
predictions = model.predict([X1_test, X2_test])
import matplotlib.pyplot as plt

def plot_pairs(X1, X2, labels, predictions, num_samples=6):
    pairs_per_row = 2
    rows = (num_samples + pairs_per_row - 1) // pairs_per_row

    plt.figure(figsize=(8, 4 * rows))

    for i in range(num_samples):
        row = i // pairs_per_row
        col_pair = i % pairs_per_row

        # Base column index (each pair uses 2 columns)
        col = col_pair * 2

        # Distance + label
        dist = float(predictions[i])
        label = labels[i]

        # Add text above the pair
        plt.subplot(rows, pairs_per_row * 2, row * pairs_per_row * 2 + col + 1)
        plt.text(
            0.5, 1.2,
            f"Label: {label} | Distance: {dist:.3f}",
            ha='center',
            va='bottom',
            transform=plt.gca().transAxes,
            fontsize=10
        )
        plt.imshow(X1[i].reshape(112, 92), cmap='gray')
        plt.axis('off')

        # Second image (same pair)
        plt.subplot(rows, pairs_per_row * 2, row * pairs_per_row * 2 + col + 2)
        plt.imshow(X2[i].reshape(112, 92), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

plot_pairs(X1_test, X2_test, y_test, predictions, num_samples=2)