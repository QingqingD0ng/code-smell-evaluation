import tensorflow as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def task_func(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    model = keras.models.Sequential([
        keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
    ])

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01))

    model.fit(X_train, Y_train, epochs=100, verbose=0)

    y_prob = model.predict(X_test).ravel()

    fpr, tpr, _ = roc_curve(Y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    return model, plt