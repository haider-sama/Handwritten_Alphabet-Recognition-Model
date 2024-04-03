import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

# Define the path to the directory containing the image dataset
dataset_path = "dataset_edited/"

# Function to load images from directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Load images and corresponding labels
X_C = load_images_from_folder(os.path.join(dataset_path, "C"))
X_F = load_images_from_folder(os.path.join(dataset_path, "F"))
X_H = load_images_from_folder(os.path.join(dataset_path, "H"))
X_L = load_images_from_folder(os.path.join(dataset_path, "L"))
X_N = load_images_from_folder(os.path.join(dataset_path, "N"))

# Combine all images and labels
X = np.concatenate((X_C, X_F, X_H, X_L, X_N), axis=0)
y = np.array([0]*len(X_C) + [1]*len(X_F) + [2]*len(X_H) + [3]*len(X_L) + [4]*len(X_N))

# Shuffle the data
X, y = shuffle(X, y)

# Define the mapping of class indices to letters
class_to_letter = {0: 'C', 1: 'F', 2: 'H', 3: 'L', 4: 'N'}

# Define the number of folds for cross-validation
num_folds = 5

# Initialize KFold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store accuracy and loss values for each fold
fold_train_accuracy = []
fold_val_accuracy = []
fold_train_loss = []
fold_val_loss = []

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{num_folds}:")
    
    # Split data into training and validation sets for this fold
    train_X, val_X = X[train_index], X[val_index]
    train_y, val_y = y[train_index], y[val_index]
    
    # Reshape the data to fit the CNN model
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], val_X.shape[2], 1)
    
    # Convert labels to one-hot encoding
    train_yOHE = to_categorical(train_y, num_classes=5)
    val_yOHE = to_categorical(val_y, num_classes=5)
    
    # Define the CNN model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(train_X.shape[1], train_X.shape[2], 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    
    # Train the model
    history = model.fit(train_X, train_yOHE, epochs=10, batch_size=32, verbose=1, callbacks=[reduce_lr, early_stop], validation_data=(val_X, val_yOHE))
    
    # Evaluate the model on training and validation data for this fold
    train_loss, train_accuracy = model.evaluate(train_X, train_yOHE, verbose=0)
    val_loss, val_accuracy = model.evaluate(val_X, val_yOHE, verbose=0)
    
    # Append accuracy and loss values to lists
    fold_train_accuracy.append(train_accuracy)
    fold_val_accuracy.append(val_accuracy)
    fold_train_loss.append(train_loss)
    fold_val_loss.append(val_loss)
    
    # Print fold results
    print(f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
    print()

# Calculate average accuracy and loss across folds
avg_train_accuracy = np.mean(fold_train_accuracy)
avg_val_accuracy = np.mean(fold_val_accuracy)
avg_train_loss = np.mean(fold_train_loss)
avg_val_loss = np.mean(fold_val_loss)

# Print average results
print("Average Train Accuracy:", avg_train_accuracy)
print("Average Val Accuracy:", avg_val_accuracy)
print("Average Train Loss:", avg_train_loss)
print("Average Val Loss:", avg_val_loss)