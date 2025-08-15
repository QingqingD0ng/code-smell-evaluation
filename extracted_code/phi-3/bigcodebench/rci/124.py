import random
import time
import matplotlib.pyplot as plt

def validate_my_list(my_list):
    if not isinstance(my_list, list):
        raise ValueError("my_list must be a list")
    if not all(isinstance(x, (int, float)) for x in my_list):
        raise ValueError("my_list must contain only numeric elements")

def calculate_total_sum(my_list):
    return sum(my_list)

def generate_random_numbers(total_sum, size):
    return [random.randint(1, min(100, total_sum)) for _ in range(size)]

def plot_histogram(random_list, size):
    plt.hist(random_list, bins=range(1, 102), edgecolor='black')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Numbers')

def enhanced_list_and_histogram(my_list, size=100, seed=100):
    validate_my_list(my_list)
    start_time = time.time()
    
    my_list.append(12)
    total_sum = calculate_total_sum(my_list)
    random_list = generate_random_numbers(total_sum, size)
    
    end_time = time.time()
    time_taken = end_time - start_time

    ax = plot_histogram(random_list, size)

    return time_taken, ax

# Example usage
try:
    my_list = [2, 3, 5]
    time_taken, ax = enhanced_list_and_histogram(my_list)
    print(f"Time taken: {time_taken} seconds")
    plt.show()
except ValueError as e:
    print(f"Error: {e}")