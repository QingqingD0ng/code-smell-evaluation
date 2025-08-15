import random as random_seed
import time
import matplotlib.pyplot as plt

def task_func(my_list, size=100, seed=100):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be a list")
    if not all(isinstance(item, (int, float)) for item in my_list):
        raise ValueError("All elements in my_list must be numeric (int or float)")
    
    my_list.append(12)
    sum_elements = sum(my_list)
    num_integers = min(size, sum_elements)
    
    random_seed.seed(seed)
    random_numbers = [random_seed.randint(1, 100) for _ in range(num_integers)]
    
    start_time = time.time()
    ax = plt.hist(random_numbers, bins=range(1, 102), edgecolor='black')
    time_taken = time.time() - start_time
    
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')
    
    return time_taken, ax