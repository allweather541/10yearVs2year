import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
csv_file_path = '/workspaces/10yearVs2year/data/T10Y2Y.csv' # Replace with your file path
# **Important:** Replace these with the actual column names in your CSV
date_column_name = 'observation_date'
value_column_name = 'T10Y2Y' # The column with the 10yr-2yr difference

# --- Step 1 & 2: Load Data and Set Index ---
try:
    # Read CSV, parse the date column automatically, and set it as the index
    df = pd.read_csv(
        csv_file_path,
        parse_dates=[date_column_name], # Tell pandas to treat this column as dates
        index_col=date_column_name      # Set this column as the DataFrame index
    )
    print(f"Successfully loaded CSV and set '{date_column_name}' as index.")

    # Optional: Sort index just in case dates are out of order in the CSV
    df.sort_index(inplace=True)

    # Optional: Rename the value column if needed for clarity
    # df.rename(columns={'OLD_COLUMN_NAME': value_column_name}, inplace=True)

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    exit()
except KeyError:
    print(f"Error: Make sure '{date_column_name}' and potentially other column names")
    print("match exactly what's in your CSV file header.")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# --- Step 3: Basic Time Series Exploration ---

print("\n--- Data Info ---")
# .info() now shows the DateTimeIndex
print(df.info())

print(f"\n--- First 5 Rows ({value_column_name}) ---")
print(df.head())

print(f"\n--- Last 5 Rows ({value_column_name}) ---")
print(df.tail())

print("\n--- Missing Values Count ---")
# Check for missing values in the yield difference column
print(df[value_column_name].isnull().sum())
# Note: Missing dates require different checks (e.g., checking frequency)

print("\n--- Descriptive Statistics ---")
print(df[value_column_name].describe())

# --- Step 3: Visualize the Time Series ---
print("\n--- Plotting the Time Series ---")
try:
    plt.figure(figsize=(12, 6)) # Set the figure size
    plt.plot(df.index, df[value_column_name], label=f'{value_column_name}')
    plt.title(f'{value_column_name} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Yield Difference')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()
    print("Plot displayed. Close the plot window to continue.")
except Exception as e:
    print(f"An error occurred during plotting: {e}")


# --- Implementation Notes ---
# 1.  **CRITICAL:** Replace `'your_data.csv'`, `'observed_date'`, and `'yield_difference'`
#     with your actual file path and column names. Case sensitivity matters!
# 2.  `parse_dates=[date_column_name]` tells pandas to try and convert that column into datetime objects.
# 3.  `index_col=date_column_name` makes the date the primary way to access rows, which is ideal for time series.
# 4.  `df.sort_index(inplace=True)` ensures the data is chronological.
# 5.  The plot is essential. Look for:
#     * **Trend:** Is the difference generally increasing, decreasing, or staying level over the long term?
#     * **Seasonality:** Are there repeating patterns within specific time periods (e.g., yearly)? (Less likely for yield curves, but possible).
#     * **Volatility:** Does the amount of fluctuation change over time?
#     * **Breaks:** Are there sudden jumps or drops?
#     * **Missing Data:** Are there gaps in the line? (We also checked with `isnull().sum()`).

from statsmodels.tsa.stattools import adfuller # Import the ADF test

# --- Assume 'df' is the DataFrame loaded and indexed from the previous step ---
# Make sure these column names match your data
# value_column_name = 'yield_difference' # Defined in the previous code block

# --- Step 4a: Handle Missing Values (Example using Forward Fill) ---
initial_missing = df[value_column_name].isnull().sum()
if initial_missing > 0:
    print(f"\n--- Handling {initial_missing} Missing Value(s) ---")
    # Forward fill: Propagates the last valid observation forward
    df[value_column_name].fillna(method='ffill', inplace=True)

    # Optional: Backward fill for any remaining NaNs at the beginning
    df[value_column_name].fillna(method='bfill', inplace=True)

    final_missing = df[value_column_name].isnull().sum()
    if final_missing == 0:
        print("Missing values handled using forward/backward fill.")
    else:
        print(f"Warning: {final_missing} missing values remain. Further investigation needed.")
else:
    print("\n--- No Missing Values Found ---")

# --- Step 4b: Check Stationarity using Augmented Dickey-Fuller (ADF) Test ---
print(f"\n--- Checking Stationarity of '{value_column_name}' ---")

# Define a function to perform and interpret the ADF test
def perform_adf_test(series, series_name):
    """Performs ADF test and prints the results."""
    print(f"ADF Test for: {series_name}")
    # The .dropna() is important if differencing created NaNs
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')

    # Interpretation
    if result[1] <= 0.05:
        print(f"Result: Evidence against the null hypothesis (p <= 0.05). The '{series_name}' series is likely stationary.")
        return True # Indicates stationarity
    else:
        print(f"Result: Weak evidence against null hypothesis (p > 0.05). The '{series_name}' series is likely non-stationary.")
        return False # Indicates non-stationarity

# Perform the test on the original (potentially filled) series
is_stationary = perform_adf_test(df[value_column_name], value_column_name)

# --- Step 4c: Apply Differencing if Necessary ---
# Many financial time series are non-stationary and require differencing.
if not is_stationary:
    print(f"\n--- Applying First-Order Differencing as series appears non-stationary ---")
    # Create a new column for the differenced data
    differenced_column_name = f'{value_column_name}_diff'
    df[differenced_column_name] = df[value_column_name].diff()

    # The first value will be NaN after differencing, so we check stationarity
    # on the series excluding the first NaN value.
    print(f"\n--- Checking Stationarity of Differenced Series '{differenced_column_name}' ---")
    # Re-run the ADF test on the differenced series
    is_diff_stationary = perform_adf_test(df[differenced_column_name], differenced_column_name)

    if not is_diff_stationary:
        print(f"\nWarning: The first-differenced series '{differenced_column_name}' still appears non-stationary.")
        print("You might need to apply second-order differencing (df[value_column_name].diff().diff())")
        print("or consider other transformations (e.g., logging) or models that handle non-stationarity.")
else:
    print(f"\n--- Original series '{value_column_name}' appears stationary. Differencing not applied. ---")


# --- Implementation Notes ---
# 1.  **Missing Values:** Forward fill (`ffill`) is common for time series, assuming the value remains constant until the next observation. If you have large gaps or specific reasons, interpolation (`df[value_column_name].interpolate(inplace=True)`) might be better. We added a backward fill (`bfill`) just in case the *very first* value(s) were missing.
# 2.  **ADF Test:**
#     * **Null Hypothesis (H0):** The time series has a unit root (it is non-stationary).
#     * **Alternative Hypothesis (H1):** The time series does not have a unit root (it is stationary).
#     * **Interpretation:** We look at the p-value. If the p-value is less than a threshold (commonly 0.05), we reject the null hypothesis and conclude the series is likely stationary. If the p-value is greater than 0.05, we fail to reject the null hypothesis and conclude the series is likely non-stationary.
# 3.  **Differencing:** `.diff()` calculates the difference between consecutive observations (`value[t] - value[t-1]`). This often helps stabilize the mean of a time series, making it stationary. The first value after differencing will always be NaN.
# 4.  **Which series to use?** If the original series (`value_column_name`) was stationary, you'll typically model that directly. If it wasn't, but the *differenced* series (`differenced_column_name`) *is* stationary, you'll often build your model (like ARIMA) based on the differenced data. The model then needs to be configured to understand it's working with differenced data (e.g., the 'd' parameter in ARIMA).