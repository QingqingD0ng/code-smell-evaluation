import random
import time
import matplotlib.pyplot as plt

def task_func(my_list, size=100, seed=100):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be a list")
    if not all(isinstance(item, (int, float)) for item in my_list):
        raise ValueError("my_list must contain only numeric values")

    my_list.append(12)
    total_sum = sum(my_list)
    random.seed(seed)
    numbers_list = [random.randint(1, min(100, total_sum)) for _ in range(size)]
    start_time = time.time()
    plt.hist(numbers_list, bins=range(1, max(numbers_list) + 2), align='left', rwidth=0.8)
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')
    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken, plt.gcf()