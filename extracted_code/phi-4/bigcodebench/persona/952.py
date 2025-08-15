import pandas as pd
import random
from datetime import datetime

def task_func(task_list, n_tasks, employees=["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"], seed=None):
    if n_tasks < 0:
        raise ValueError("Number of tasks cannot be negative.")
    
    if seed is not None:
        random.seed(seed)
    
    due_date = datetime.now().strftime('%Y-%m-%d')
    
    tasks_to_assign = task_list * (n_tasks // len(task_list)) + task_list[:n_tasks % len(task_list)]
    assigned_tasks = random.sample(tasks_to_assign, n_tasks)
    
    assigned_employees = [random.choice(employees) for _ in range(n_tasks)]
    
    sanitized_tasks = [task.replace(" ", "_") for task in assigned_tasks]
    
    df = pd.DataFrame({
        'Task Name': sanitized_tasks,
        'Assigned To': assigned_employees,
        'Due Date': [due_date] * n_tasks
    })
    
    return df