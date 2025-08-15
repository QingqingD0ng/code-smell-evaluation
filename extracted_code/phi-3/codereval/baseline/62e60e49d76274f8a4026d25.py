class UnitOfWork:
    def __init__(self, metadata=None, timeout=None):
        self.metadata = metadata if metadata is not None else {}
        self.timeout = timeout

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Simulate the processing time of the function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Check if the execution time exceeded the timeout
            if self.timeout is not None and (end_time - start_time) > self.timeout:
                raise TimeoutError(f"Function exceeded timeout of {self.timeout} seconds.")
            
            return result
        
        # Attach metadata and timeout to the decorator
        wrapper.metadata = self.metadata
        wrapper.timeout = self.timeout
        return wrapper