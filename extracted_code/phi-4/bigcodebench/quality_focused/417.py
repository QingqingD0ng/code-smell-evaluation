import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def task_func(X, Y):
    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    
    # Create a Sequential model
    model = Sequential()
    
    # Add a dense hidden layer with sigmoid activation
    model.add(Dense(8, input_dim=2, activation='sigmoid'))
    
    # Add output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
    
    # Fit the model
    history = model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=0, validation_data=(X_test, Y_test))
    
    # Plot training and validation loss
    ax = plt.figure().gca()
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    return model, ax