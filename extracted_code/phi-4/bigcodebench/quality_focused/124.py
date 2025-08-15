from random import randint, seed as random_seed
import time
import matplotlib.pyplot as plt

def task_func(my_list, size=100, seed=100):
    if not isinstance(my_list, list):
        raise TypeError("Input must be a list.")
    if not all(isinstance(i, (int, float)) for i in my_list):
        raise ValueError("All elements in the list must be numeric.")

    my_list.append(12)
    random_seed(seed)
    total_sum = sum(my_list)
    num_elements = min(total_sum, size)
    random_numbers = [randint(1, 100) for _ in range(num_elements)]

    start_time = time.time()
    plt.figure()
    plt.hist(random_numbers, bins=range(1, 102), edgecolor='black')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')
    ax = plt.gca()
    time_taken = time.time() - start_time

    return time_taken, ax