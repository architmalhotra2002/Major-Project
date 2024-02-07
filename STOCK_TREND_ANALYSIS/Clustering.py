import pandas_datareader as web
import pandas as pd
import numpy as np
import datetime as dt
from yahoo_fin import stock_info as si

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Get Dow Jones 30 companies
companies_dow30 = si.tickers_dow()

clusters = 5

start = dt.datetime.now() - dt.timedelta(days=365 * 2)
end = dt.datetime.now()

# Create an empty DataFrame to store stock data
data = pd.DataFrame(columns=companies_dow30)

# Fetch data for Dow Jones 30 companies
for ticker in companies_dow30:
    try:
        ticker_data = web.DataReader(ticker, 'yahoo', start, end)
        data[ticker] = ticker_data['Close']
    except Exception as e:
        print(f"Failed to retrieve data for {ticker}: {str(e)}")

# Drop any rows with missing values (NaN)
data.dropna(inplace=True)

# Calculate daily percentage changes
percentage_changes = data.pct_change().fillna(0)

# Standardize data
scaler = StandardScaler()
percentage_changes_scaled = scaler.fit_transform(percentage_changes)

# KMeans Clustering
clustering_model = KMeans(n_clusters=clusters, max_iter=1000, random_state=42)
clusters = clustering_model.fit_predict(percentage_changes_scaled)

# Add clustering results to DataFrame
results = pd.DataFrame({
    'tickers': companies_dow30,
    'clusters': clusters
})

# Sort DataFrame by cluster labels
results = results.sort_values(by=['clusters'], axis=0)

print(results)
