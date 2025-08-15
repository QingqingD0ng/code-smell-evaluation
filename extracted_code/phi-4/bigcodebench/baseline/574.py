from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def task_func(array_length=100, noise_level=0.2):
    x = np.linspace(0, 4 * np.pi, array_length)
    y = np.sin(x) + noise_level * np.random.normal(size=array_length)

    def model(x, a, b, c):
        return a * np.sin(b * x + c)

    params, _ = curve_fit(model, x, y, p0=[1, 1, 0])

    y_fit = model(x, *params)

    ax = plt.figure().gca()
    ax.plot(x, y, 'b.', label='Noisy Data')
    ax.plot(x, y_fit, 'r-', label='Fitted Curve')
    plt.legend()
    plt.show()

    return ax