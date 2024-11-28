import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
MODEL_PATH = "models/histopathology_model.h5"
DATA_PATH = "data/"

# Load data
def load_data():
    # Example: Replace with actual dataset loading logic
    test_data = np.load(os.path.join(DATA_PATH, "test_images.npy"))
    test_labels = np.load(os.path.join(DATA_PATH, "test_labels.npy"))
    return test_data, test_labels

# Evaluate model
def evaluate_model():
    test_data, test_labels = load_data()

    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Predict and evaluate
    predictions = np.argmax(model.predict(test_data), axis=1)
    cm = confusion_matrix(test_labels, predictions)
    print("Classification Report:\n", classification_report(test_labels, predictions))

    # Visualise confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
