import numpy as np
import matplotlib.pyplot as plt

def task_func(seed=42, image_size=(100, 100, 3), range_low=0, range_high=255):
    if range_low >= range_high:
        raise ValueError("range_low must be less than range_high")
    
    random.seed(seed)
    image = np.random.randint(range_low, range_high, size=image_size, dtype=np.uint8)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    return None, image