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

class AlphabetRecognizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_to_letter = {0: 'C', 1: 'F', 2: 'H', 3: 'L', 4: 'N'} # Dictionary for the alphabets indexes. Initialized as classes.
        self.classes = ['C', 'F', 'H', 'L', 'N'] # Class labels
        self.num_elements = [] # Number of elements per class
        self.num_images = [] # Number of images per class

    def load_images_from_folder(self, folder):
        # Load images from a directory
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        return images

    def load_dataset(self):
        # Load the dataset and labels
        X_C = self.load_images_from_folder(os.path.join(self.dataset_path, "C"))
        X_F = self.load_images_from_folder(os.path.join(self.dataset_path, "F"))
        X_H = self.load_images_from_folder(os.path.join(self.dataset_path, "H"))
        X_L = self.load_images_from_folder(os.path.join(self.dataset_path, "L"))
        X_N = self.load_images_from_folder(os.path.join(self.dataset_path, "N"))

        # Calculate number of elements and images per class
        self.num_elements = [len(X_C), len(X_F), len(X_H), len(X_L), len(X_N)]
        self.num_images = [len(X_C), len(X_F), len(X_H), len(X_L), len(X_N)]

        # Combine all images and corresponding labels
        X = np.concatenate((X_C, X_F, X_H, X_L, X_N), axis=0)
        y = np.array([0]*len(X_C) + [1]*len(X_F) + [2]*len(X_H) + [3]*len(X_L) + [4]*len(X_N))

        self.X, self.y = shuffle(X, y)

    def split_dataset(self, test_size=0.2, random_state=42):
        # Split the dataset into training and testing sets
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

    def preprocess_data(self):
        # Preprocessing data
        # Reshape the data to fit into the CNN Model
        self.train_X = self.train_X.reshape(self.train_X.shape[0], self.train_X.shape[1], self.train_X.shape[2], 1)
        self.test_X = self.test_X.reshape(self.test_X.shape[0], self.test_X.shape[1], self.test_X.shape[2], 1)
        # Convert labels to one-hot encoding
        self.train_yOHE = to_categorical(self.train_y, num_classes=5)
        self.test_yOHE = to_categorical(self.test_y, num_classes=5)

    def build_model(self):
        # Build CNN Model
        # Using two Droput layers with a value of 0.4 and L2 Regularization to reduce & avoid overfitting.
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(
            self.train_X.shape[1], self.train_X.shape[2], 1)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu',
                             kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128, activation='relu',
                             kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(5, activation='softmax'))

    def compile_model(self, learning_rate=0.001):
        # Compiling the model using Adam Optimizer
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, epochs=5, callbacks=None):
        # Train the CNN Model
        # Initial Epochs = 5; use Epochs = 10 for greater accuracy
        if callbacks is None:
            callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001),
                         EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')]
        self.history = self.model.fit(self.train_X, self.train_yOHE, epochs=epochs, callbacks=callbacks,
                                       validation_data=(self.test_X, self.test_yOHE))

    def calculate_f1_score(self):
        # Calculate F1 Score
        y_true = np.argmax(self.test_yOHE, axis=1)
        y_pred = np.argmax(self.model.predict(self.test_X), axis=1)
        self.f1score = f1_score(y_true, y_pred, average='macro')

    def plot_training_history(self):
        # Plot Training/Validation Accuracy using Matplotlib
        plt.plot(self.history.history['accuracy'], label='Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val_Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        

    def predict_external_image(self, external_img_path):
        # Testing the model on external image 
        # Predicting the class and letter of the respective external image
        img = cv2.imread(external_img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img, (1, 28, 28, 1))
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)
        predicted_letter = self.class_to_letter[predicted_class]
        return predicted_class, predicted_letter

# Main Function
def main():
    dataset_path = "dataset_edited/"
    image_classifier = AlphabetRecognizer(dataset_path)
    image_classifier.load_dataset()
    image_classifier.split_dataset()
    image_classifier.preprocess_data()
    image_classifier.build_model()
    image_classifier.compile_model()
    image_classifier.train_model()
    image_classifier.calculate_f1_score()
    image_classifier.plot_training_history()
    
    # Directory of the external image
    external_img_path = "test_images/L.png"
    predicted_class, predicted_letter = image_classifier.predict_external_image(external_img_path)
    print("Predicted Class for External Image:", predicted_class)
    print("Predicted Letter for External Image:", predicted_letter)
    
if __name__ == "__main__":
    main()