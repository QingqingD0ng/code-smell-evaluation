import pandas as pd
import random
from datetime import datetime

def task_func(task_list, n_tasks, employees=["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"], seed=None):
    if n_tasks < 0:
        raise ValueError("n_tasks cannot be negative")
    
    if seed is not None:
        random.seed(seed)
    
    task_list = [task.replace(" ", "_") for task in task_list]
    assigned_tasks = []
    
    for _ in range(min(n_tasks, len(task_list))):
        task = task_list.pop(random.randrange(len(task_list)))
        employee = random.choice(employees)
        due_date = datetime.now().strftime("%Y-%m-%d")
        assigned_tasks.append((task, employee, due_date))
    
    df = pd.DataFrame(assigned_tasks, columns=['Task Name', 'Assigned To', 'Due Date'])
    return df