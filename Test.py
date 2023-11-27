import yfinance as yf
import spacy
from alpha_vantage.timeseries import TimeSeries

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def get_stock_data(query, api_key):
    # Process the user query using spaCy
    doc = nlp(query)

    # Extract relevant information from the query
    organization_name = None

    # Look for the first entity with label "ORG" in the spaCy processed query
    for ent in doc.ents:
        if ent.label_ == "ORG":
            organization_name = ent.text
            break

    print(f"Organization Name: {organization_name}")

    if organization_name:
        stock_symbol = get_stock_symbol(organization_name, api_key)
        if stock_symbol:
            try:
                # Query current stock data using yfinance
                stock_data = yf.download(stock_symbol)

                if not stock_data.empty:
                    return stock_data
                else:
                    print(f"No stock data found for {organization_name}.")
                    return None
            except Exception as e:
                print(f"Error querying stock data from Yahoo Finance: {e}")
                return None
        else:
            print(f"No stock symbol found for {organization_name}.")
            return None
    else:
        print("No organization name found in the query.")
        return None

def get_stock_symbol(organization_name, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')

    try:
        # Search for the organization's stock symbol using the Alpha Vantage API
        data, meta_data = ts.get_symbol_search(keywords=organization_name)

        # Extract the stock symbol from the search results
        stock_symbol = data.index[0]

        if stock_symbol:
            return stock_symbol
        else:
            print(f"No stock symbol found for {organization_name}.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example query and API key
user_query = "Tell me about the performance of Apple stock."
alpha_vantage_api_key = 'T1JWEZ4MU5R5BCWN'

# Get current stock data based on the user query
result = get_stock_data(user_query, alpha_vantage_api_key)

# Process and present the result as needed
if result is not None:
    print("Current Stock Data:")
    print(result.head())
else:
    print("Could not extract stock information from the query.")
