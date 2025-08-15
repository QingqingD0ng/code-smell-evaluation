import pandas as pd
import random
from datetime import datetime

def task_func(
    task_list,
    n_tasks,
    employees=None,
    seed=None
):
    if n_tasks < 0:
        raise ValueError("n_tasks cannot be negative")
    if employees is None:
        employees = ["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"]
    random.seed(seed)
    task_list = [task.replace(' ', '_') for task in task_list]
    assigned_tasks = []
    for _ in range(min(n_tasks, len(task_list))):
        task = random.choice(task_list)
        employee = random.choice(employees)
        due_date = datetime.now().date()
        assigned_tasks.append((task, employee, due_date))
        task_list.remove(task)  # Ensure task is not reassigned
    return pd.DataFrame(assigned_tasks, columns=['Task Name', 'Assigned To', 'Due Date'])

# Example usage:
df = task_func(['Clean Office', 'Prepare Report', 'Client Meeting'], 2, seed=42)
print(df)
print(type(df))