import pandas as pd
import random
from datetime import datetime

def task_func(task_list, n_tasks, employees=["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"], seed=None):
    if n_tasks < 0:
        raise ValueError("Number of tasks cannot be negative")

    if seed is not None:
        random.seed(seed)

    due_date = datetime.now().strftime("%Y-%m-%d")

    if n_tasks > len(task_list):
        task_list = task_list * (n_tasks // len(task_list)) + task_list[:n_tasks % len(task_list)]
    else:
        task_list = random.sample(task_list, n_tasks)

    task_assignments = [
        {
            "Task Name": task.replace(" ", "_"),
            "Assigned To": random.choice(employees),
            "Due Date": due_date
        }
        for task in task_list
    ]

    return pd.DataFrame(task_assignments)