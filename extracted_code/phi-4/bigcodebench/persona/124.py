from random import randint, seed as random_seed
import time
import matplotlib.pyplot as plt

def task_func(my_list, size=100, seed=100):
    if not isinstance(my_list, list):
        raise TypeError("Input must be a list.")
    if not all(isinstance(x, (int, float)) for x in my_list):
        raise ValueError("All elements in the list must be numeric.")

    my_list.append(12)
    random_seed(seed)
    sum_of_elements = sum(my_list)
    list_size = min(sum_of_elements, size)
    
    start_time = time.time()
    random_numbers = [randint(1, 100) for _ in range(list_size)]
    time_taken = time.time() - start_time
    
    plt.figure()
    ax = plt.hist(random_numbers, bins=range(1, 102), edgecolor='black', align='left')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')
    
    return time_taken, ax

# Example usage:
# my_list = [2, 3, 5]
# time_taken, ax = task_func(my_list)
# print(type(time_taken))  # Example output: <class 'float'>
# print(ax.get_title())    # Returns 'Histogram of Random Numbers'