import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Any

def extract_numeric_values(original: List[Tuple[str, Any]]) -> np.ndarray:
    """Extract numeric values from a list of tuples."""
    return np.array([t[1] for t in original if isinstance(t[1], (int, float))])

def compute_statistics(arr: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics for a numpy array."""
    return {
      'mean': np.mean(arr),
      'std': np.std(arr),
      'min': np.min(arr),
      'max': np.max(arr)
    }

def plot_histogram_and_pdf(arr: np.ndarray, ax: Any) -> None:
    """Plot a histogram with overlaid probability density function."""
    mu, sigma = stats.norm.fit(arr)
    n, bins, patches = ax.hist(arr, density=True, alpha=0.6, bins='auto', label='Histogram')
    y = stats.norm.pdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--', linewidth=2, label='PDF')
    ax.legend()

def task_func(original: List[Tuple[str, Any]]) -> Tuple[np.ndarray, Dict[str, float], Any]:
    """Perform data extraction, statistical analysis, and plotting."""
    arr = extract_numeric_values(original)
    stats = compute_statistics(arr)
    fig, ax = plt.subplots()
    plot_histogram_and_pdf(arr, ax)
    return arr, stats, ax