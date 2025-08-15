from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, title, xlabel, ylabel):
    ax = plt.gca()
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

def train_and_evaluate_model(X, Y):
    if X.shape[1]!= 2 or len(np.unique(Y))!= 2:
        raise ValueError('Input data X must have 2 columns and target labels Y must have 2 unique values.')
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    model = Sequential([
        Dense(16, activation='sigmoid', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    sgd = SGD(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=0)
    
    plot_training_history(history, 'Model Loss', 'Epoch', 'Loss')
    return model