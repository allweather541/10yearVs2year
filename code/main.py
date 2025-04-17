# Import the pandas library
import pandas as pd

# --- Configuration ---
# Replace 'your_data.csv' with the actual path to your CSV file
csv_file_path = 'your_data.csv'

# --- Load the Data ---
try:
    # Read the CSV file into a pandas DataFrame
    # A DataFrame is like a table, perfect for structured data
    df = pd.read_csv(csv_file_path)

    print("Successfully loaded the CSV file!")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please make sure the file path is correct and the file exists.")
    # Exit or handle the error appropriately if the file isn't found
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    # Exit or handle other potential errors
    exit()

# --- Implementation Notes ---
# 1. Make sure the python script and your CSV file are in the same directory,
#    or provide the full path to the CSV file (e.g., 'C:/Users/YourUser/Documents/data/your_data.csv').
# 2. Pandas offers many options for read_csv (e.g., specifying separators, handling headers).
#    Check the documentation if your CSV is non-standard:
#    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html