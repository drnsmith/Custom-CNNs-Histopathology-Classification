import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle
import os

# Paths
DATA_PATH = "data/"
MODEL_SAVE_PATH = "models/histopathology_model.h5"

# Load and pre-process data
def load_data():
    # Example: Replace with actual dataset loading logic
    # Load training and validation data
    train_data = np.load(os.path.join(DATA_PATH, "train_images.npy"))
    train_labels = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    val_data = np.load(os.path.join(DATA_PATH, "val_images.npy"))
    val_labels = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

    return (train_data, train_labels), (val_data, val_labels)

def preprocess_data(train_data, train_labels):
    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )
    datagen.fit(train_data)
    return datagen.flow(train_data, train_labels, batch_size=32)

# Build CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main training logic
def train_model():
    (train_data, train_labels), (val_data, val_labels) = load_data()
    train_gen = preprocess_data(train_data, train_labels)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))

    # Build and train the model
    model = build_model(input_shape=train_data.shape[1:])
    model.fit(
        train_gen,
        validation_data=(val_data, val_labels),
        epochs=50,
        class_weight=class_weights
    )

    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
