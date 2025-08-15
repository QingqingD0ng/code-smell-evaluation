from random import randint, seed as random_seed
import time
import matplotlib.pyplot as plt

def task_func(my_list, size=100, seed=100):
    if not isinstance(my_list, list):
        raise TypeError("Input must be a list.")
    if not all(isinstance(x, (int, float)) for x in my_list):
        raise ValueError("All elements in the list must be numeric.")

    my_list.append(12)
    total_sum = sum(my_list)
    num_elements = min(total_sum, size)

    random_seed(seed)
    random_numbers = [randint(1, 100) for _ in range(num_elements)]

    start_time = time.time()
    plt.figure()
    ax = plt.hist(random_numbers, bins=range(1, 102), edgecolor='black', align='left')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')
    plt.close()

    end_time = time.time()
    time_taken = end_time - start_time

    return time_taken, ax