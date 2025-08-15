import random
import time
import matplotlib.pyplot as plt

def task_func(my_list, size=100, seed=100):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be a list")
    if not all(isinstance(x, (int, float)) for x in my_list):
        raise ValueError("my_list must contain only numeric elements")

    random_seed(seed)
    start_time = time.time()

    my_list.append(12)
    total_sum = sum(my_list)
    random_list = [random.randint(1, min(100, total_sum)) for _ in range(size)]
    
    end_time = time.time()
    time_taken = end_time - start_time

    plt.hist(random_list, bins=range(1, 102), edgecolor='black')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')

    return time_taken, plt.gcf().axes[0]