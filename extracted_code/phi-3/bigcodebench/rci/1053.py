import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
    - file_path (str): The path to the input CSV file.

    Returns:
    - pandas.DataFrame: The DataFrame containing the CSV data.

    Raises:
    - FileNotFoundError: If the specified file_path does not exist. It raises a FileNotFoundError with a message indicating the file path that was not found.
    - pd.errors.EmptyDataError: If the CSV file is empty. It raises a ValueError indicating that the CSV file is empty.
    - Exception: For any other errors that occur during the function execution. In this case, the error is printed to the console, and None is returned.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("CSV file is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")
    except Exception as e:
        print(e)
        return None

def process_text_data(df, save_path=None):
    """
    Processes text data in a DataFrame and generates a histogram of the top ten most common words.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the text data.
    - save_path (str, optional): The path where the histogram plot will be saved. If not provided, the plot is displayed on the screen.

    Returns:
    - matplotlib.axes.Axes: The Axes object of the plot if save_path is not provided.
    Useful for further customization or display in notebooks.
    - None: If save_path is provided, the plot is saved to the