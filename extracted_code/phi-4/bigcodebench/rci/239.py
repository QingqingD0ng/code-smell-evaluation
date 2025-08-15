import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def extract_numeric_values(original):
    return np.array([item[1] for item in original])

def compute_statistics(numeric_values):
    return {
       'mean': np.mean(numeric_values),
       'std': np.std(numeric_values),
       'min': np.min(numeric_values),
       'max': np.max(numeric_values)
    }

def plot_histogram_with_pdf(numeric_values, ax, plot_color='r-', line_width=2):
    counts, bins, _ = ax.hist(numeric_values, density=True, alpha=0.6, bins='auto')
    mu, sigma = stats.norm.fit(numeric_values)
    pdf_x = np.linspace(np.min(numeric_values), np.max(numeric_values), 100)
    pdf_y = stats.norm.pdf(pdf_x, mu, sigma)
    ax.plot(pdf_x, pdf_y, plot_color, lw=line_width)

def task_func(original):
    numeric_values = extract_numeric_values(original)
    statistics = compute_statistics(numeric_values)
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plot_histogram_with_pdf(numeric_values, ax)
    
    return numeric_values, statistics, ax