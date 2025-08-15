from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def task_func(X, Y, learning_rate=0.01):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    model = Sequential([
        Dense(16, activation='sigmoid', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')

    history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), verbose=0)
    
    ax = plt.subplot(1, 1, 1)
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    return model, ax