import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def task_func(size=1000):
    data = np.random.normal(size=size)
    mean, std = np.mean(data), np.std(data)
    pdf_x = np.linspace(mean - 4*std, mean + 4*std, 100)
    pdf_y = stats.norm.pdf(pdf_x, mean, std)
    
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.6, color='g')
    ax.plot(pdf_x, pdf_y, 'r-', lw=2)
    return fig