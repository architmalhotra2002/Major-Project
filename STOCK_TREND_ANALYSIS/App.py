import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import yfinance as yf
from yahoo_fin import stock_info
import pyttsx3

def Say(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    
    # Adjust voice selection based on available voices
    engine.setProperty('voice', voices[0].id)  # You can change [0] to select a different voice
    engine.setProperty('rate', 170)  # Adjust the speech rate as needed
    
    # Print the spoken text
    print("A.I:", Text)
    
    # Speak the text
    engine.say(Text)
    engine.runAndWait()

start = '2010-01-01'
end = '2023-11-20'

st.title('Stock Trend Prediction')


start = '2010-01-01'
end = '2019-12-31'
user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start=start, end=end)


def get_stock_summary(ticker):
    try:
        summary_data = stock_info.get_quote_table(ticker)
        summary_paragraph = "\n".join([f"{key}: {value}" for key, value in summary_data.items()])
        return summary_paragraph
    except Exception as e:
        return f"Error: {e}"

# Example: Get summary for Apple Inc. (AAPL)
ticker_symbol = "AAPL"
stock_summary = get_stock_summary(ticker_symbol)
st.write(stock_summary)


#Describing Data

st.subheader('Data from 2010-2023')
st.write(df.describe())


#Visualizations

st.subheader('Closing Price vs Time Chart')

fig = plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

#splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)


# Loading the Model
model = load_model('keras_model.h5')

#Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test , y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1 /scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor 

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Assuming y_predicted and y_test are arrays of predicted and actual prices

# Identify uptrends and downtrends
trend_threshold = 0.02  # Adjust this threshold based on your data and model performance

uptrends = y_predicted > (1 + trend_threshold) * y_test
downtrends = y_predicted < (1 - trend_threshold) * y_test
# Count the occurrences of uptrends and downtrends
uptrend_count = np.count_nonzero(uptrends)
downtrend_count = np.count_nonzero(downtrends)

total_samples = len(uptrends)

# Count the occurrences of uptrends and downtrends
uptrend_count = np.count_nonzero(uptrends)
downtrend_count = np.count_nonzero(downtrends)

total_samples = len(uptrends)

# Calculate the percentage of uptrends and downtrends
uptrend_percentage = (uptrend_count / total_samples) * 100
downtrend_percentage = (downtrend_count / total_samples) * 100

# Convert percentages to strings
uptrend_percentage_str = "{:.2f}%".format(uptrend_percentage)
downtrend_percentage_str = "{:.2f}%".format(downtrend_percentage)

# Generate summary paragraphs
uptrend_summary = f"The model predicts an uptrend in {uptrend_percentage_str} of the cases."
downtrend_summary = f"The model predicts a downtrend in {downtrend_percentage_str} of the cases."

# Display summaries
st.subheader('Trend Summary')
st.write(uptrend_summary)
st.write(downtrend_summary)
Say(uptrend_summary)
Say(downtrend_summary)
