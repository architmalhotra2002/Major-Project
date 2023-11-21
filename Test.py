import yfinance as yf
import spacy
from dateutil.relativedelta import relativedelta
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
from datetime import datetime

def get_stock_data(query):
    # Process the user query using spaCy
    doc = nlp(query)

    # Extract relevant information from the query
    stock_symbol = None
    duration = None

    for ent in doc.ents:
        if ent.label_ == "ORG":  # Assume stock symbols are entities of type "ORG"
            stock_symbol = get_stock_symbol(ent.text, 'T1JWEZ4MU5R5BCWN')
        elif ent.label_ == "DATE":  # Assume duration is specified as a date entity
            duration = ent.text

    # Set default start date to 1 year ago
    start_date = (datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')

    # Calculate end date based on the specified duration
    if duration:
        if "month" in duration.lower():
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = start_date
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if stock_symbol:
        try:
            # Convert the start and end date strings to datetime objects using datetime.fromisoformat()
            start_date = datetime.fromisoformat(start_date)
            end_date = datetime.fromisoformat(end_date)

            print(f"After conversion - start_date: {start_date}, end_date: {end_date}")

            # Query stock data using yfinance
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
            return stock_data
        except Exception as e:
            print(f"Error converting dates to datetime objects: {e}")
            return None
    else:
        return None

def get_stock_symbol(company_name, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')

    try:
        # Search for the company's stock symbol using the Alpha Vantage API
        data, meta_data = ts.get_symbol_search(keywords=company_name)

        # Extract the stock symbol from the search results
        stock_symbol = data.index[0]

        if stock_symbol:
            return stock_symbol
        else:
            print(f"No stock symbol found for {company_name}.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example query

user_query = "Tell me about the performance of Apple stock in the last month."

# Get stock data based on the user query
result = get_stock_data(user_query)

# Process and present the result as needed
if result is not None:
    print("Stock Data:")
    print(result.head())
else:
    print("Could not extract stock information from the query.")
