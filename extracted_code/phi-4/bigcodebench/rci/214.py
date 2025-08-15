import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def task_func(seed=42, image_size=(100, 100, 3), range_low=0, range_high=255):
    if range_low >= range_high:
        raise ValueError("range_low must be less than range_high")
    
    random.seed(seed)
    np.random.seed(seed)
    image = np.random.randint(range_low, range_high, image_size, dtype=np.uint8)
    
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    ax = plt.gca()
    plt.show()
    
    return ax, image