from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def task_func(X, Y, hidden_neurons=10, learning_rate=0.01, epochs=10, test_size=0.3, random_state=42, verbose=0):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    model = keras.models.Sequential([
        keras.layers.Dense(hidden_neurons, activation='sigmoid', input_shape=(2,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=epochs, verbose=verbose)

    Y_pred = model.predict(X_test).ravel()
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC curve')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right')

    return model, ax