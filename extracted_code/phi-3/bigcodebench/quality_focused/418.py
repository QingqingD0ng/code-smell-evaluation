import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def task_func(X, Y):
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Create a Keras Sequential model
    model = tf.keras.Sequential([
        layers.Dense(10, activation='sigmoid', input_shape=(2,)),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(X_train, Y_train, epochs=100, verbose=0, batch_size=1)

    # Predict probabilities for the test set
    Y_pred_probs = model.predict(X_test)

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(Y_test.ravel(), Y_pred_probs.ravel())
    auc_score = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC: {auc_score:.2f}')
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')

    return model, plt

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y =