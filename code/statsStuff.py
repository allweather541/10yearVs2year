import pandas as pd
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