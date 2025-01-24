import pandas as pd
import plotly.express as px
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
# Define URLs for training and test data
train_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

# Load the training dataset
train_df = pd.read_csv(train_data_url)

# Load the test dataset
test_df = pd.read_csv(test_data_url)

# Convert 'Timestamp' columns to datetime
train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])

# Set 'Timestamp' as the index
train_df.set_index('Timestamp', inplace=True)
test_df.set_index('Timestamp', inplace=True)

# Plot the training data
px.line(train_df, x=train_df.index, y='trips', title="Hourly Taxi Trips (Training Data)")
# Define the model using Exponential Smoothing
model = ExponentialSmoothing(
    train_df['trips'],          # Target variable
    trend="add",                # Additive trend
    seasonal="add",             # Additive seasonality
    seasonal_periods=24         # 24-hour daily seasonality
)
# Fit the model
modelFit = model.fit()

# Print the model summary
print(modelFit.summary())
# Forecast for the test period (744 hours in January)
pred = modelFit.forecast(steps=744)

# Convert predictions into a DataFrame
forecast_df = pd.DataFrame({
    'Timestamp': pd.date_range(start='2019-01-01', periods=744, freq='H'),
    'Forecasted Trips': pred
})
# Combine actual and forecasted data
comparison_df = pd.DataFrame({
    'Actual Trips': test_df['trips'].values,
    'Forecasted Trips': pred.values
}, index=test_df.index)

# Plot actual vs forecasted trips
px.line(comparison_df, title="Actual vs Forecasted Taxi Trips")

