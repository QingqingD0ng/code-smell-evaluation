class UnitOfWorkDecorator:
    def __init__(self, metadata=None, timeout=None):
        self.metadata = metadata
        self.timeout = timeout

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TimeoutError:
                if self.timeout:
                    raise TimeoutError(f"Function exceeded timeout of {self.timeout} seconds.")
                else:
                    raise
        wrapper.metadata = self.metadata
        wrapper.timeout = self.timeout
        return wrapper

def unit_of_work(metadata=None, timeout=None):
    return UnitOfWorkDecorator(metadata, timeout)