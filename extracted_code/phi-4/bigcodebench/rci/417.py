from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

def task_func(X, Y, learning_rate=0.01, epochs=100, hidden_units=4, test_size=0.25, verbose=0):
    if X.shape[1]!= 2:
        raise ValueError("Input dimension must be 2.")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = Sequential()
    model.add(Dense(hidden_units, input_dim=2, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, verbose=verbose)

    ax = plt.subplot(1, 1, 1)
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    return model, ax