
'''
Main code for training a Siamese neural network for face recognition
'''
import utils
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
test_pairs, test_labels, test_pair_subjects = utils.create_pairs(X_test, y_test, 5, return_subject_ids=True)
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
    epochs=15,  # more epochs
    validation_data=([X1_test, X2_test], test_labels)
)
model.evaluate([X1_test, X2_test], test_labels)


#------------------------------evaluation metrics
predictions = model.predict([X1_test, X2_test])

# choose a threshold for the distance to represent a match/non-match
threshold = .8
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

error_indices = np.where(predicted_labels != true_labels)[0]
error_count = len(error_indices)
error_subjects = Counter()
for idx in error_indices:
    s1, s2 = test_pair_subjects[idx]
    if true_labels[idx] == 1:
        # same-subject pair labeled as different
        error_subjects[int(s1)] += 1
    else:
        # different-subject pair labeled as same
        error_subjects[int(s1)] += 1
        error_subjects[int(s2)] += 1

print(f"\nTotal incorrect pairs: {error_count} / {len(true_labels)}")
print("Subject error counts:")
for subject, count in error_subjects.most_common():
    print(f"  Subject {subject}: {count} errors")

if error_subjects:
    top_subject, top_count = error_subjects.most_common(1)[0]
    print(f"\nMost frequently involved subject in errors: {top_subject} ({top_count} times)")


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


def plot_subject_images(subject_ids, X, y, max_images=9):
    for subject_id in subject_ids:
        original_id = int(subject_id + 35)
        subject_images = X[y == subject_id]
        if len(subject_images) == 0:
            print(f"No images found for subject {subject_id}")
            continue

        num_images = min(len(subject_images), max_images)
        fig, axes = plt.subplots(1, num_images, figsize=(2 * num_images, 2))
        if num_images == 1:
            axes = [axes]

        for i in range(num_images):
            axes[i].imshow(subject_images[i].reshape(112, 92), cmap='gray')
            axes[i].axis('off')
        fig.suptitle(f"Subject {subject_id} (original folder s{original_id})")
        plt.show()


def plot_incorrect_pairs_for_subjects(subject_ids, X1, X2, pair_subjects, true_labels, predicted_labels, num_pairs=4):
    matching_error_indices = []
    for idx in np.where(predicted_labels != true_labels)[0]:
        s1, s2 = pair_subjects[idx]
        if int(s1) in subject_ids or int(s2) in subject_ids:
            matching_error_indices.append(idx)

    if not matching_error_indices:
        print("No incorrect pairs found for the selected subjects.")
        return

    sample_indices = matching_error_indices[:num_pairs]
    num_plots = len(sample_indices)
    fig, axes = plt.subplots(num_plots, 2, figsize=(6, 3 * num_plots))
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    for ax_row, idx in zip(axes, sample_indices):
        s1, s2 = pair_subjects[idx]
        label = true_labels[idx]
        pred = predicted_labels[idx]

        ax_row[0].imshow(X1[idx].reshape(112, 92), cmap='gray')
        ax_row[0].set_title(f"Subj {int(s1)}")
        ax_row[0].axis('off')

        ax_row[1].imshow(X2[idx].reshape(112, 92), cmap='gray')
        ax_row[1].set_title(f"Subj {int(s2)}")
        ax_row[1].axis('off')

        fig.suptitle(f"Incorrect pair idx {idx}: true={label} pred={pred}")

    plt.tight_layout()
    plt.show()

selected_subjects = [1, 2, 4]
print(f"\nShowing sample images for subjects: {selected_subjects}")
plot_subject_images(selected_subjects, X_test, y_test, max_images=9)
print("\nShowing some incorrect pairs for these subjects:")
plot_incorrect_pairs_for_subjects(selected_subjects, X1_test, X2_test, test_pair_subjects, true_labels, predicted_labels, num_pairs=4)

# Find subjects with zero errors
perfect_subjects = []
for subject_id in range(1, 6):
    if subject_id not in error_subjects:
        perfect_subjects.append(subject_id)

print(f"\nSubjects with zero errors: {perfect_subjects[:2]}")


def plot_all_subjects_combined(error_subjects_list, perfect_subjects_list, X, y, max_per_subject=3):
    """Display error and perfect subjects in one figure"""
    all_subjects = error_subjects_list + perfect_subjects_list
    num_subjects = len(all_subjects)
    
    fig, axes = plt.subplots(num_subjects, max_per_subject, figsize=(8, 3 * num_subjects))
    if num_subjects == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for row, subject_id in enumerate(all_subjects):
        subject_images = X[y == subject_id]
        num_images = min(len(subject_images), max_per_subject)
        
        for col in range(max_per_subject):
            ax = axes[row, col]
            if col < num_images:
                ax.imshow(subject_images[col].reshape(112, 92), cmap='gray')
                if col == 0:
                    original_id = int(subject_id + 35)
                    if subject_id in error_subjects_list:
                        ax.set_ylabel(f"Subject {subject_id}\n(s{original_id})\n[ERROR]", fontweight='bold')
                    else:
                        ax.set_ylabel(f"Subject {subject_id}\n(s{original_id})\n[PERFECT]", fontweight='bold', color='green')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_all_subjects_combined(selected_subjects, perfect_subjects[:2], X_test, y_test, max_per_subject=3)


def plot_correct_pairs(X1, X2, true_labels, predicted_labels, pair_subjects, num_same=3, num_different=3):
    """Show correctly classified pairs (same and different)"""
    correct_indices = np.where(predicted_labels == true_labels)[0]
    
    # Separate same-class and different-class correct pairs
    correct_same = [idx for idx in correct_indices if true_labels[idx] == 1]
    correct_different = [idx for idx in correct_indices if true_labels[idx] == 0]
    
    # Random selection
    np.random.seed(42)
    same_samples = np.random.choice(correct_same, min(num_same, len(correct_same)), replace=False)
    diff_samples = np.random.choice(correct_different, min(num_different, len(correct_different)), replace=False)
    
    total_pairs = len(same_samples) + len(diff_samples)
    
    fig, axes = plt.subplots(total_pairs, 2, figsize=(6, 3 * total_pairs))
    if total_pairs == 1:
        axes = np.expand_dims(axes, axis=0)
    
    row = 0
    # Plot correct same pairs
    for idx in same_samples:
        s1, s2 = pair_subjects[idx]
        axes[row, 0].imshow(X1[idx].reshape(112, 92), cmap='gray')
        axes[row, 0].set_title(f"Subj {int(s1)}")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(X2[idx].reshape(112, 92), cmap='gray')
        axes[row, 1].set_title(f"Subj {int(s2)}")
        axes[row, 1].axis('off')
        
        fig.text(0.02, 0.5 + (total_pairs - row - 1) * (.8 / total_pairs), 
                 "SAME\n(correct)", rotation=90, verticalalignment='center', fontweight='bold', color='green')
        row += 1
    
    # Plot correct different pairs
    for idx in diff_samples:
        s1, s2 = pair_subjects[idx]
        axes[row, 0].imshow(X1[idx].reshape(112, 92), cmap='gray')
        axes[row, 0].set_title(f"Subj {int(s1)}")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(X2[idx].reshape(112, 92), cmap='gray')
        axes[row, 1].set_title(f"Subj {int(s2)}")
        axes[row, 1].axis('off')
        
        fig.text(0.02, 0.5 + (total_pairs - row - 1) * (.8 / total_pairs), 
                 "DIFFERENT\n(correct)", rotation=90, verticalalignment='center', fontweight='bold', color='blue')
        row += 1
    
    plt.suptitle("Correctly Classified Pairs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#print("\nShowing correctly classified pairs (same and different):")
#plot_correct_pairs(X1_test, X2_test, true_labels, predicted_labels, test_pair_subjects, num_same=3, num_different=3)


def plot_single_positive_negative(X1, X2, predictions, true_labels, predicted_labels, pair_subjects):
    """Display one correct positive and one correct negative pair with their scores"""
    correct_indices = np.where(predicted_labels == true_labels)[0]
    
    # Find one correct same pair (label=1)
    correct_same = [idx for idx in correct_indices if true_labels[idx] == 1]
    # Find one correct different pair (label=0)
    correct_different = [idx for idx in correct_indices if true_labels[idx] == 0]
    
    if not correct_same or not correct_different:
        print("Not enough correct pairs to display")
        return
    
    np.random.seed(42)
    same_idx = np.random.choice(correct_same)
    diff_idx = np.random.choice(correct_different)
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    # Positive pair (same person)
    s1, s2 = pair_subjects[same_idx]
    score_same = float(predictions[same_idx])
    
    axes[0].imshow(X1[same_idx].reshape(112, 92), cmap='gray')
    axes[0].set_title(f"Subject {int(s1)}", fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(X2[same_idx].reshape(112, 92), cmap='gray')
    axes[1].set_title(f"Subject {int(s2)}", fontsize=10)
    axes[1].axis('off')
    
    fig.text(0.245, 0.95, f"Distance: {score_same:.4f}", ha='center', fontsize=12, fontweight='bold', color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    fig.text(0.245, 0.02, "SAME (Correct)", ha='center', fontsize=11, fontweight='bold', color='green')
    
    # Negative pair (different people)
    s1, s2 = pair_subjects[diff_idx]
    score_diff = float(predictions[diff_idx])
    
    axes[2].imshow(X1[diff_idx].reshape(112, 92), cmap='gray')
    axes[2].set_title(f"Subject {int(s1)}", fontsize=10)
    axes[2].axis('off')
    
    axes[3].imshow(X2[diff_idx].reshape(112, 92), cmap='gray')
    axes[3].set_title(f"Subject {int(s2)}", fontsize=10)
    axes[3].axis('off')
    
    fig.text(0.755, 0.95, f"Distance: {score_diff:.4f}", ha='center', fontsize=12, fontweight='bold', color='blue', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    fig.text(0.755, 0.02, "DIFFERENT (Correct)", ha='center', fontsize=11, fontweight='bold', color='blue')
    
    plt.suptitle("Correct Predictions: Positive vs Negative", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.10)
    plt.show()

plot_single_positive_negative(X1_test, X2_test, predictions, true_labels, predicted_labels, test_pair_subjects)


def plot_predictions_distribution(predictions, true_labels):
    """Show distribution of distances for positive and negative pairs"""
    predictions_flat = predictions.reshape(-1)
    
    # Separate by true label
    positive_dists = predictions_flat[true_labels == 1]
    negative_dists = predictions_flat[true_labels == 0]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(positive_dists, bins=30, alpha=0.6, label='Same Person (Positive)', color='green', edgecolor='black')
    axes[0, 0].hist(negative_dists, bins=30, alpha=0.6, label='Different Person (Negative)', color='red', edgecolor='black')
    axes[0, 0].axvline(.8, color='blue', linestyle='--', linewidth=2, label='Decision Threshold (.8)')
    axes[0, 0].set_xlabel('Distance Score', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Distance Scores', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot([positive_dists, negative_dists], labels=['Same Person', 'Different Person'], patch_artist=True)
    axes[0, 1].set_ylabel('Distance Score', fontsize=11)
    axes[0, 1].set_title('Box Plot: Score Ranges', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Scatter plot
    positive_indices = np.where(true_labels == 1)[0]
    negative_indices = np.where(true_labels == 0)[0]
    axes[1, 0].scatter(positive_indices, positive_dists, alpha=0.5, s=20, color='green', label='Same Person')
    axes[1, 0].scatter(negative_indices, negative_dists, alpha=0.5, s=20, color='red', label='Different Person')
    axes[1, 0].axhline(.8, color='blue', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[1, 0].set_xlabel('Pair Index', fontsize=11)
    axes[1, 0].set_ylabel('Distance Score', fontsize=11)
    axes[1, 0].set_title('Distance Scores by Pair Index', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Statistics table
    axes[1, 1].axis('off')
    stats_text = f"""
STATISTICS

SAME PERSON (Positive) Pairs:
  Count: {len(positive_dists)}
  Mean: {np.mean(positive_dists):.4f}
  Std Dev: {np.std(positive_dists):.4f}
  Min: {np.min(positive_dists):.4f}
  Max: {np.max(positive_dists):.4f}
  Median: {np.median(positive_dists):.4f}

DIFFERENT PERSON (Negative) Pairs:
  Count: {len(negative_dists)}
  Mean: {np.mean(negative_dists):.4f}
  Std Dev: {np.std(negative_dists):.4f}
  Min: {np.min(negative_dists):.4f}
  Max: {np.max(negative_dists):.4f}
  Median: {np.median(negative_dists):.4f}

THRESHOLD: 0.5
  Positive pairs < 0.5: {np.sum(positive_dists < .8)} / {len(positive_dists)}
  Negative pairs >= 0.5: {np.sum(negative_dists >= .8)} / {len(negative_dists)}
    """
    axes[1, 1].text(0.1, .8, stats_text, fontsize=10, family='monospace', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Siamese Network Distance Score Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("\nShowing distance score distributions:")
plot_predictions_distribution(predictions, true_labels)


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


