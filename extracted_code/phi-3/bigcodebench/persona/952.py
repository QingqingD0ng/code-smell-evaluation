import pandas as pd
import random
from datetime import datetime


def task_func(
    task_list,
    n_tasks,
    employees=["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"],
    seed=None,
):
    if n_tasks < 0:
        raise ValueError("n_tasks must be non-negative")
    
    if seed is not None:
        random.seed(seed)
    
    sanitized_tasks = [task.replace(" ", "_") for task in task_list]
    if n_tasks > len(sanitized_tasks):
        raise ValueError("Cannot assign more tasks than available")
    
    assigned_tasks = random.sample(list(zip(sanitized_tasks, employees)), n_tasks)
    
    return pd.DataFrame(assigned_tasks, columns=['Task Name', 'Assigned To'])