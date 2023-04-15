# Stock-Price-Prediction-with-LSTM

LSTM stands for Long Short Term Memory Networks. It is a type of recurrent neural network that is commonly used for regression and time series forecasting in machine learning. It can memorize data for long periods, which differentiates LSTM neural networks from other neural networks. If you want to learn how to predict stock prices with LSTM, this article is for you. In this article, I will walk you through the task of stock price prediction with LSTM using Python

Using LSTM is one of the best machine learning approaches for time series forecasting. LSTMs are recurrent neural networks designed to remember data for a longer period. So, whenever you are working on a problem where your neural network fails to memorize data, you can use LSTM neural network. You can read more about LSTMs

```
import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2
data = yf.download('AAPL', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", 
             "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
data.tail()

```
A candlestick chart gives a clear picture of the increase and decrease in stock prices, so let’s visualize a candlestick chart of the data before moving further:

```
import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = "Apple Stock Price Analysis", 
                     xaxis_rangeslider_visible=False)
figure.show()
```

![image](https://user-images.githubusercontent.com/85225054/232210638-a3cbab88-1bbd-41f8-871c-2c1a6bfaf73d.png)

Now let’s have a look at the correlation of all the columns with the Close column as it is the target column:

```
import seaborn as sns
correlation = data.corr()
# print(correlation["Close"].sort_values(ascending=False))
sns.heatmap(correlation, annot=True)
```
## Training LSTM for Stock Price Prediction
Now I will start with training an LSTM model for predicting stock prices. I will first split the data into training and test sets:

```
x = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
```

Now we will prepare a neural network architecture for LSTM:

```
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

```
Now here is how we can train our neural network model for stock price prediction:
```
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)
```
Now let’s test this model by giving input values according to the features that we have used to train this model and predicting the final result:

```
import numpy as np
#features = [Open, High, Low, Adj Close, Volume]
features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
model.predict(features)
```

Now Save the model 
```
import pickle
pickle.dump(model, open("model.pkl", "wb"))
model.save("model.h5")

```



