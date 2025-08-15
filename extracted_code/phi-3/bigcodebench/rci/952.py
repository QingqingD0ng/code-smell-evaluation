import pandas as pd
import random
from datetime import datetime
from typing import List, Optional

def sanitize_task_name(task: str) -> str:
    return task.replace(" ", "_")

def task_func(
    task_list: List[str],
    n_tasks: int,
    employees: Optional[List[str]] = None,
    seed: Optional[int] = None,
    max_tasks_per_employee: Optional[int] = None,
) -> pd.DataFrame:
    if n_tasks < 0:
        raise ValueError("n_tasks cannot be negative")

    if seed is not None:
        random.seed(seed)

    if employees is None:
        employees = ["John Doe", "Jane Smith", "James Brown", "Mary Johnson", "Robert Davis"]

    if n_tasks > len(task_list):
        raise ValueError("n_tasks cannot be greater than the length of task_list")

    if max_tasks_per_employee is not None and max_tasks_per_employee < 0:
        raise ValueError("max_tasks_per_employee cannot be negative")

    # Sanitize task names
    task_list = [sanitize_task_name(task) for task in task_list]

    # Initialize data to be used for DataFrame
    assigned_tasks = []

    # Assign tasks to employees
    for _ in range(n_tasks):
        # If max_tasks_per_employee is set, get a list of eligible employees
        if max_tasks_per_employee is not None:
            eligible_employees = [emp for emp in employees if assigned_tasks.count((emp, None, None)) < max_tasks_per_employee]
        else:
            eligible_employees = employees

        # Assign task to a random eligible employee
        employee = random.choice(eligible_employees)
        task = random.choice(task_list)
        assigned_tasks.append([task, employee, datetime.now().date()])

        # Remove assigned task from task list
        task_list.remove