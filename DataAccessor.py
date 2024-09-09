import pandas as pd
from datetime import datetime

class DataAccessor:
    def __init__(self, data, separator=','):
        """
        Initialize the DataAccessor with a file-like object, CSV file path, or a pandas DataFrame.
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data, sep=separator)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        elif hasattr(data, 'read'):
            self.df = pd.read_csv(data, sep=separator)
        else:
            raise ValueError("Input must be a CSV file path, a pandas DataFrame, or a file-like object")

    def set_timestamp_column(self, column_name):
        """
        Set the specified column as the timestamp column and convert it to datetime.
        """
        try:
            self.df[column_name] = pd.to_datetime(self.df[column_name], unit='ms', errors='coerce')
        except Exception as e:
            print(f"Error converting column {column_name} to datetime: {e}")

    def get_column_info(self):
        """
        Return a summary of the DataFrame's columns and their data types.
        """
        info = {}
        for column in self.df.columns:
            info[column] = str(self.df[column].dtype)
        return info

    def update_column_type(self, column_name, new_type):
        """
        Update the data type of a specified column.
        """
        try:
            if new_type == 'datetime':
                self.df[column_name] = pd.to_datetime(self.df[column_name], errors='coerce')
            elif new_type == 'numeric':
                self.df[column_name] = pd.to_numeric(self.df[column_name], errors='coerce')
            elif new_type == 'string':
                self.df[column_name] = self.df[column_name].astype(str)
            else:
                raise ValueError("Unsupported data type specified")
        except Exception as e:
            print(f"Error updating column type for {column_name} to {new_type}: {e}")

    def overview(self):
        """
        Provide a structured overview of the extracted column information.
        """
        column_info = self.get_column_info()
        overview_list = []
        for col, dtype in column_info.items():
            overview_list.append({"Column Name": col, "Data Type": dtype})
        return pd.DataFrame(overview_list)
