from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

def task_func(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    model = Sequential([
        Dense(16, activation='sigmoid', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    sgd = SGD(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=0)
    ax = plt.gca()
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    return model, ax