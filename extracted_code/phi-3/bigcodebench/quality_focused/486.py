import matplotlib.pyplot as plt

def task_func(start_time, end_time, step, trend, seed=42):
    if step <= 0:
        raise ValueError("Step must be greater than 0")
    if end_time <= start_time:
        raise ValueError("End time must be greater than start time")
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_time, end=end_time, freq='{}ms'.format(step))
    
    # Generate random values from a normal distribution
    values = np.random.randn(len(timestamps))
    
    # Add a linear trend based on the provided trend value
    values += np.arange(len(timestamps)) * trend
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Time': timestamps,
        'Value': values
    })
    
    # Plot the time series
    ax = data.plot(x='Time', y='Value', kind='line')
    
    # Set the x-axis label
    ax.set_xlabel('Time')
    
    # Set the y-axis label
    ax.set_ylabel('Value')
    
    return ax

# Example usage:
# ax = task_func(0, 10000, 100, 0.001)
# print(ax.get_xticklabels())