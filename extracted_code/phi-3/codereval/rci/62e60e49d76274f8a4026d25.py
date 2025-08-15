from functools import wraps
import time

class UnitOfWorkMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        metadata = kwargs.pop('metadata', {})
        timeout = kwargs.pop('timeout', None)
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                except TimeoutError:
                    raise TimeoutError(f"Function exceeded timeout of {timeout} seconds.")
                end_time = time.time()
                if timeout is not None and (end_time - start_time) > timeout:
                    raise TimeoutError(f"Function exceeded timeout of {timeout} seconds.")
                wrapper.metadata = metadata
                wrapper.timeout = timeout
                return result
            return wrapper
        
        return decorator

    def __call__(cls, func):
        return cls.decorator(func)

class UnitOfWork(metaclass=UnitOfWorkMeta):
    pass

def my_function_with_work_unit(a, b):
    # Simulate some processing time
    time.sleep(1)
    return a + b

# Set metadata and timeout
@UnitOfWork(metadata={'unit': 'work'}, timeout=2)
def my_function_with_work_unit_2(*args, **kwargs):
    return my_function_with_work_unit(*args, **kwargs)

# Usage
try:
    result = my_function_with_work_unit_2(2, 3)
    print(result)
except TimeoutError as e:
    print(e)