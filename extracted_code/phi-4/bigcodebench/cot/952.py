import pandas as pd
import random
from datetime import datetime

def task_func(task_list, n_tasks, employees=["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"], seed=None):
    if n_tasks < 0:
        raise ValueError("n_tasks cannot be negative.")

    if seed is not None:
        random.seed(seed)

    current_date = datetime.now().strftime('%Y-%m-%d')
    tasks_to_assign = task_list * (n_tasks // len(task_list) + 1)
    random.shuffle(tasks_to_assign)
    tasks_to_assign = tasks_to_assign[:n_tasks]

    assignments = []
    for task in tasks_to_assign:
        employee = random.choice(employees)
        sanitized_task = task.replace(" ", "_")
        assignments.append({
            "Task Name": sanitized_task,
            "Assigned To": employee,
            "Due Date": current_date
        })

    return pd.DataFrame(assignments)