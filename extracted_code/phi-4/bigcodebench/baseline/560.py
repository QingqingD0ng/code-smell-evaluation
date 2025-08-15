from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def task_func(data):
    # Convert the data string into a list of dictionaries
    data_list = [dict(zip(['date', 'value'], item.split('-'))) for item in data.split(',')]
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract the year and month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    
    # Filter for a single year, assuming all data is from the same year
    year = df['year'].iloc[0]
    df = df[df['year'] == year]
    
    # Set'month' as the index for plotting
    df.set_index('month', inplace=True)
    
    # Convert 'value' to numeric
    df['value'] = pd.to_numeric(df['value'])
    
    # Plot the bar chart
    ax = df['value'].plot(kind='bar', figsize=(10, 6))
    
    # Set title and labels
    ax.set_title(f'Monthly Data for {year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return ax