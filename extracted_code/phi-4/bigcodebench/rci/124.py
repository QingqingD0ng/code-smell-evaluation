from random import randint, seed as random_seed
import time
import matplotlib.pyplot as plt

MIN_RANDOM = 1
MAX_RANDOM = 100
BIN_RANGE = MAX_RANDOM + 1

def generate_random_numbers(sum_elements, max_size, seed):
    num_elements = min(max_size, sum_elements)
    random_seed(seed)
    return [randint(MIN_RANDOM, MAX_RANDOM) for _ in range(num_elements)]

def plot_histogram(random_numbers):
    plt.hist(random_numbers, bins=range(MIN_RANDOM, BIN_RANGE), edgecolor='black')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')
    plt.show()

def task_func(my_list, size=100, seed=100):
    if not isinstance(my_list, list):
        raise TypeError("Input must be a list.")
    if not all(isinstance(x, (int, float)) for x in my_list):
        raise ValueError("All elements in the list must be numeric.")

    my_list.append(12)
    total_sum = sum(my_list)

    start_time = time.time()
    random_numbers = generate_random_numbers(total_sum, size, seed)
    end_time = time.time()

    time_taken = end_time - start_time
    plot_histogram(random_numbers)
    return time_taken, plt.gca()