
'''
Main code for training a Siamese neural network for face recognition
'''
import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
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

history = model.fit(
    [X1_train, X2_train],
    train_labels,
    batch_size=10,
    epochs=3,  # more epochs
    validation_data=([X1_test, X2_test], test_labels)
)
model.evaluate([X1_test, X2_test], test_labels)


#------------------------------evaluation metrics
predictions = model.predict([X1_test, X2_test])

# choose a threshold for the distance to represent a match/non-match
threshold = 0.5
predicted_labels = (predictions < threshold).astype(int).reshape(-1)
true_labels = test_labels.reshape(-1)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=['Different', 'Same']))
print("Confusion Matrix:")
print(cm)


def plot_confusion_matrix(cm, classes=['Different', 'Same'], normalize=False, title='Confusion Matrix'):
    plt.figure(figsize=(5, 4))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm)


#------------------------------visual stuff

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

#plot_pairs(X1_test, X2_test, y_test, predictions, num_samples=2)






  


