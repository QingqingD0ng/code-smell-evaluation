from random import seed as random_seed, randint

import time

import matplotlib.pyplot as plt


def task_func(my_list, size=100, seed=100):

    if not isinstance(my_list, list):

        raise TypeError("my_list must be a list")

    if not all(isinstance(item, (int, float)) for item in my_list):

        raise ValueError("my_list must contain only numeric elements")


    my_list.append(12)

    total_sum = sum(my_list)

    random_seed(seed)

    generated_numbers = [randint(1, 100) for _ in range(min(size, int(total_sum)))]


    start_time = time.time()

    plt.hist(generated_numbers, bins=10, range=(1, 101), edgecolor='black')

    plt.xlabel('Number')

    plt.ylabel('Frequency')

    plt.title('Histogram of Random Numbers')

    ax = plt.gca()


    time_taken = time.time() - start_time

    return time_taken, ax