import pandas as pd
import random
from datetime import datetime

def task_func(task_list, n_tasks, employees=["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"], seed=None):
    if n_tasks < 0:
        raise ValueError("Number of tasks cannot be negative.")
    
    if seed is not None:
        random.seed(seed)
    
    due_date = datetime.now().strftime("%Y-%m-%d")
    
    task_assignments = []
    task_names = [task.replace(" ", "_") for task in task_list]
    
    for i in range(n_tasks):
        task = random.choice(task_names)
        employee = random.choice(employees)
        task_assignments.append((task, employee, due_date))
    
    df = pd.DataFrame(task_assignments, columns=["Task Name", "Assigned To", "Due Date"])
    
    return df