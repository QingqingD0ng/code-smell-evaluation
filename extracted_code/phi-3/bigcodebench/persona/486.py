import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, HourLocator, SecondLocator

def task_func(start_time, end_time, step, trend, seed=42):
    np.random.seed(seed)
    time_range = pd.date_range(start=pd.Timestamp(datetime.fromtimestamp(start_time / 1000)), end=pd.Timestamp(datetime.fromtimestamp(end_time / 1000)), freq=f'{step}ms')
    values = np.random.normal(loc=0, scale=1, size=len(time_range)) + trend * np.arange(len(time_range))
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(time_range, values)
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(HourLocator())
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    plt.show()
    return ax