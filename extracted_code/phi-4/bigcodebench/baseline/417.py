from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

def task_func(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(4, input_dim=2, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=0)

    ax = plt.subplot()
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    return model, ax