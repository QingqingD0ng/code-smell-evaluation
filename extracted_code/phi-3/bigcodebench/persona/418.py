from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def task_func(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = keras.models.Sequential([
        keras.layers.Dense(1, input_dim=2, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=0.01))
    model.fit(X_train, Y_train, epochs=100, verbose=0)
    y_pred = model.predict(X_test).ravel()
    fpr, tpr, _ = roc_curve(Y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    ax = plt.gca()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    return model, ax