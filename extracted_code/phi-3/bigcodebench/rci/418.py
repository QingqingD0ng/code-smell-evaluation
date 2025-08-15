import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Configuration constants
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 10

def plot_roc_curve(fpr, tpr, roc_auc, ax):
    """Plot the ROC curve on the provided Axes object."""
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc='lower right')

def validate_input_data(X, Y):
    if X.shape[1]!= 2:
        raise ValueError("Input data X must have exactly 2 features per sample.")
    if Y.shape[1]!= 1:
        raise ValueError("Target data Y must have exactly 1 target per sample.")

def task_func(X, Y, learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE):
    validate_input_data(X, Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    model = keras.Sequential([
        layers.Dense(16, input_dim=2, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),