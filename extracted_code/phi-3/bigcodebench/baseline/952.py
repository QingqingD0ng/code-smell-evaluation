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
        raise ValueError("n_tasks cannot be negative")

    if seed is not None:
        random.seed(seed)

    # Sanitize task names
    task_list = [task.replace(" ", "_") for task in task_list]

    # Prepare data for DataFrame
    tasks = [[task, random.choice(employees), datetime.now().date()] for task in task_list[:n_tasks]]

    # Create DataFrame and set column names
    df = pd.DataFrame(tasks, columns=['Task Name', 'Assigned To', 'Due Date'])

    return df