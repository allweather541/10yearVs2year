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
# --- Assume 'df' is the DataFrame processed in the previous steps ---
# --- Assume 'value_column_name' is 'T10Y2Y' ---
value_column_name = 'T10Y2Y' # Make sure this matches your column

# --- Step 5: Split the Data Chronologically ---

# Calculate the index number for the split point (80% of the data)
split_point = int(len(df) * 0.8)

# --- FIX: Use .iloc for integer-based slicing ---
# Select rows up to (but not including) split_point for training
train_series = df[value_column_name].iloc[:split_point]
# Select rows from split_point to the end for testing
test_series = df[value_column_name].iloc[split_point:]
# --- End of FIX ---

# Verify the split
print("\n--- Data Splitting ---")
print(f"Total observations: {len(df)}")
print(f"Training observations: {len(train_series)} (Index from {train_series.index.min()} to {train_series.index.max()})")
print(f"Testing observations: {len(test_series)} (Index from {test_series.index.min()} to {test_series.index.max()})")

# Check if the lengths add up
if len(train_series) + len(test_series) == len(df):
    print("Train and test set lengths add up correctly.")
else:
    print("Warning: Train and test set lengths do not sum to the total!")

# Display the last few training points and first few testing points
print("\nLast 3 training points:")
print(train_series.tail(3))
print("\nFirst 3 testing points:")
print(test_series.head(3))

# --- Implementation Notes ---
# 1. We now slice the specific column (Series) df[value_column_name] using .iloc.
# 2. .iloc[:split_point] takes rows from the beginning up to (excluding) the integer index `split_point`.
# 3. .iloc[split_point:] takes rows from the integer index `split_point` to the end.
# 4. This method correctly uses integer positions for splitting, regardless of the index label type (like DatetimeIndex).

import pmdarima as pm # Import pmdarima

# --- Assume 'train_series' is available from the previous step ---

# --- Step 6: Find Best ARIMA Order and Fit Model ---
print("\n--- Finding Best ARIMA Model using auto_arima ---")

# Use auto_arima to find the best p, q (d=0 since data is stationary)
# We set seasonal=False as strong daily seasonality is less common in yield spreads
# trace=True shows the models being tested
# error_action='ignore', suppress_warnings=True help avoid stopping on non-converging models
try:
    auto_model = pm.auto_arima(
        train_series,
        start_p=1, start_q=1,   # Starting points for p and q search
        max_p=5, max_q=5,     # Maximum p and q to test (adjust if needed)
        d=0,                  # Integration order (0 because data is stationary)
        seasonal=False,       # No seasonality assumed for daily data
        stepwise=True,        # Use stepwise algorithm (faster)
        suppress_warnings=True,
        error_action='ignore',
        trace=True            # Print models being tested
    )

    print("\n--- Best ARIMA Model Found ---")
    # Print the summary of the best model found
    # This includes the chosen (p,d,q) order and coefficient values
    print(auto_model.summary())

    # The 'auto_model' object is now our fitted model
    fitted_model = auto_model

except ImportError:
    print("\nError: pmdarima library not found. Please install it: pip install pmdarima")
    fitted_model = None # Ensure variable exists even on error
except Exception as e:
    print(f"\nAn error occurred during auto_arima fitting: {e}")
    fitted_model = None # Ensure variable exists even on error

# --- Implementation Notes ---
# 1. Add this code block after the data splitting block in your script.
# 2. `auto_arima` will test various combinations of p and q (up to max_p, max_q)
#    and select the one with the best fit based on criteria like AIC (Akaike Information Criterion).
# 3. Setting `d=0` is crucial because we already determined the series is stationary.
# 4. `seasonal=False` is a reasonable starting assumption for daily yield spreads. If you suspected a weekly/monthly pattern, you might explore `seasonal=True` with an appropriate `m` value (e.g., m=5 or m=21 for trading days).
# 5. The `trace=True` output will show you the different ARIMA(p,0,q) models it tries and their AIC scores. The one with the lowest AIC is usually chosen.
# 6. The `auto_model.summary()` provides detailed information about the coefficients, standard errors, p-values, and diagnostic tests for the chosen model.
