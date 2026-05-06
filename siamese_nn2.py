
'''
Main code for training a Siamese neural network for face recognition
'''
import utils2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
   

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    

    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    


    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))
    return model


def siamese_nn():
    input_top = Input(shape=(112, 92, 1))
    input_bottom = Input(shape=(112, 92, 1))

    shared = shared_network()

    embedding_top = shared(input_top)
    embedding_bottom = shared(input_bottom)

    distance = Lambda(utils2.euclidean_distance)(
        [embedding_top, embedding_bottom]
    )

    model = Model(
        inputs=[input_top, input_bottom],
        outputs=distance
    )

    return model


X_train, y_train, X_test, y_test = utils2.get_data("att_faces")
y_test = y_test - 35
train_pairs, train_labels = utils2.create_pairs(X_train, y_train, 35)
test_pairs, test_labels = utils2.create_pairs(X_test, y_test, 5)
X1_train = train_pairs[:, 0]
X2_train = train_pairs[:, 1]

X1_test = test_pairs[:, 0]
X2_test = test_pairs[:, 1]

model = siamese_nn()

model.compile(
    optimizer=Adam(learning_rate=0.0003, clipnorm=1.0),
    loss=utils2.contrastive_loss,
    metrics=[utils2.accuracy]
)

history = model.fit(
    [X1_train, X2_train],
    train_labels,
    batch_size=10,
    epochs=8,  # more epochs
    validation_data=([X1_test, X2_test], test_labels)
)
model.evaluate([X1_test, X2_test], test_labels)






#------------------------------plot accuracy over epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], marker='o', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.title('Siamese Network Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_over_epochs.png')
plt.show()

#------------------------------confusion matrix
predictions = model.predict([X1_test, X2_test])
predicted_labels = (predictions < 0.5).astype('int32').reshape(-1)

cm = confusion_matrix(test_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Different', 'Same'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix for Siamese Network Test Pairs')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
