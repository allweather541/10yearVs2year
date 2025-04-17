# Import the pandas library
import pandas as pd

# --- Configuration ---
# Replace 'your_data.csv' with the actual path to your CSV file
csv_file_path = '/workspaces/10yearVs2year/data/T10Y2Y.csv'

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

# --- Explore the Data ---

# Display the first 5 rows of the DataFrame
print("\n--- First 5 Rows ---")
print(df.head())

# Display the last 5 rows of the DataFrame
print("\n--- Last 5 Rows ---")
print(df.tail())

# Get the dimensions of the DataFrame (rows, columns)
print("\n--- Shape (Rows, Columns) ---")
print(df.shape)

# Get a concise summary of the DataFrame
# Includes column names, non-null counts, and data types
print("\n--- Data Info ---")
print(df.info())

# Get descriptive statistics for numerical columns
# Includes count, mean, standard deviation, min, max, and quartiles
print("\n--- Descriptive Statistics (Numerical Columns) ---")
print(df.describe())

# Get descriptive statistics for object (e.g., string) columns
print("\n--- Descriptive Statistics (Object/Categorical Columns) ---")
print(df.describe(include='object')) # or include='all' for both

# Check for missing values in each column
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# --- Implementation Notes ---
# 1. Run these commands after successfully loading the data.
# 2. `df.head()` and `df.tail()` help you see the actual data values.
# 3. `df.shape` tells you how big your dataset is.
# 4. `df.info()` is crucial for understanding data types (like numbers, text, dates) and identifying columns with missing values immediately.
# 5. `df.describe()` gives a quick statistical overview of numerical data (like averages, ranges). `describe(include='object')` shows frequency and unique counts for text data.
# 6. `df.isnull().sum()` provides a count of missing (NaN or Null) values per column, which is vital for data cleaning later.