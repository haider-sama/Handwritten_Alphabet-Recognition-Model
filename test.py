import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from keras import regularizers


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

# Define the mapping of class indices to letters
class_to_letter = {0: 'C', 1: 'F', 2: 'H', 3: 'L', 4: 'N'}

classes = ['C', 'F', 'H', 'L', 'N']
num_elements = [len(X_C), len(X_F), len(X_H), len(X_L), len(X_N)]
num_images = [len(X_C), len(X_F), len(X_H), len(X_L), len(X_N)]

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Classes')
ax1.set_ylabel('Number of Elements', color=color)
ax1.bar(classes, num_elements, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Number of Images', color=color)  
ax2.plot(classes, num_images, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.show()

# Combine all images and labels
X = np.concatenate((X_C, X_F, X_H, X_L, X_N), axis=0)
y = np.array([0]*len(X_C) + [1]*len(X_F) + [2]*len(X_H) + [3]*len(X_L) + [4]*len(X_N))

# Shuffle the data
X, y = shuffle(X, y)

# Split the dataset into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data to fit the CNN model
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)

# Convert labels to one-hot encoding
train_yOHE = to_categorical(train_y, num_classes=5)
test_yOHE = to_categorical(test_y, num_classes=5)


# Define the CNN model with reduced complexity and increased regularization
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(train_X.shape[1], train_X.shape[2], 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# Train the model
history = model.fit(train_X, train_yOHE, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=(test_X, test_yOHE))




# Calculate F1 Score
y_true = np.argmax(test_yOHE, axis=1)
y_pred = np.argmax(model.predict(test_X), axis=1)
f1score = f1_score(y_true, y_pred, average='macro')

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val_Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction on external image
external_img_path = "L.png"
img = cv2.imread(external_img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = np.reshape(img, (1, 28, 28, 1))
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
predicted_letter = class_to_letter[predicted_class]
print("Predicted Class for External Image:", predicted_class)
print("Predicted Letter for External Image:", predicted_letter)

print("The training accuracy is :", history.history['accuracy'])
print("The training loss is :", history.history['loss'])
print("The validation accuracy is :", history.history['val_accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The F1 Score is :", f1score)