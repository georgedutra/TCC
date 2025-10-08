import time
from functools import wraps

# Global dictionary to store execution times
execution_times = {}

def timeit(func):
    """Decorator that measures the execution time of a function and stores it.
    
    The execution time can be retrieved from the global `execution_times` dictionary
    using the function name as key. Also stores in the function's `last_execution_time` attribute.
    
    Args:
        func: The function to be timed.
        
    Returns:
        The wrapped function that stores execution time.
        
    Example:
        ```
        @timeit
        def my_function():
            time.sleep(1)
        
        my_function()
        print(execution_times['my_function'])  # Access via global dict
        print(my_function.last_execution_time)  # Access via function attribute
        ```
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store in global dictionary
        execution_times[func.__name__] = execution_time
        
        # Store as function attribute
        wrapper.last_execution_time = execution_time
        
        return result
    
    # Initialize attribute
    wrapper.last_execution_time = None
    
    return wrapper