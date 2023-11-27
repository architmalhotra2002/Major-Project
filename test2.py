from google.finance import finance_service

# Set the start and end dates
start_date = "2022-01-01"
end_date = "2023-01-01"

# Define the query parameters
query = {
    "q": "TCS.NS",                           # Stock symbol
    "from": start_date,                       # Start date
    "to": end_date,                         # End date
    "output_type": "csv",                  # Output format
    "timeframe": "daily"                   # Timeframe
}

# Fetch historical stock data using Google Finance API
response = finance_service.get_historical_data(query)

# Parse the CSV response and extract the data
data = []
for row in response.splitlines():
    if len(row) > 0:
        date, open, high, low, close, volume = row.split(",")
        data.append((date, open, high, low, close, volume))

# Convert the data into a Pandas DataFrame
import pandas as pd
df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
df["Date"] = pd.to_datetime(df["Date"])

# Display the stock data
print(df)
