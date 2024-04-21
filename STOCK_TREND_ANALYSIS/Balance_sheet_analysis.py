import pandas_datareader as web
import pandas as pd
from yahoo_fin import stock_info as si
import datetime as dt

tickers = si.tickers_sp500()

start = dt.datetime.now() - dt.timedelta(days=365)
end = dt.datetime.now()

sp500_df = web.DataReader('^GSPC', 'yahoo', start, end)
sp500_df['Pct Change'] = sp500-df['Adj Close'].pct_change()
sp500_return = (sp500_df['Pct Change']+1).cumprod()[-1]

return_list = []
final_df = pd.DataFrame(columns=['Ticker', 'Lastest_Price', 'Score', 'PE_Ratio', 'PEG_Ratio', 'SMA_150', 'SMA_200', '52_Week_Low', '52_Week_High'])

for ticker in tickers:
    