#This program uses an artifical neural network called long shortime memory of lstm for sure. We are going to use this network to preduct the closing stock price of a corporation

#import the library

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime

plt.style.use('dark_background')

#get the stock quote
var = input("What is your stock?: ").upper()

now = datetime.now()
now = now.date()

#defining the data frame
df = web.DataReader(var, data_source = 'yahoo', start = '2012-01-01', end = now)

#show the data frame
print(df)

#get the number of rows and columns in the data set
print(df.shape)

#visualize the closing price history
plt.figure(figsize = (18,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel("Close Price USD ($)", fontsize = 18)
plt.show()

#create a new data frame with only the close column
data = df.filter(['Close'])

#convert the data frame to a numpy array
dataset = data.values

#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

print()

print(training_data_len)

#scale the data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

#create the training data set
#creating the scaled training data

train_data = scaled_data[0:training_data_len,:]

#split the data into x_train and y_train data sets

x_train = []
y_train = []

for x in range(60, len(train_data)):
    x_train.append(train_data[x-60:x, 0])
    y_train.append(train_data[x, 0])

    if x<= 60:
        print(x_train)
        print(y_train)
        print()

#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer = "adam", loss = "mean_squared_error")

#train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

#create the testing data set
#create a new array containing scaled values from index 1543 to 2003

test_data = scaled_data[training_data_len - 60:, :]

#create the data sets X_test and y_test
X_test = []
y_test = dataset[training_data_len:, :]

for x in range(60, len(test_data)):
    X_test.append(test_data[x-60:x, 0])

#convert the data into a numpy array
X_test = np.array(X_test)

#reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#get the models predicted price values
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

#get the root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions - y_test)**2)))

print(rmse)

#plot the data

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# #visualize the data
# plt.figure(figsize = (18,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USE ($)', fontsize = 18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
# plt.show()

#show the valid and predicted prices
print(valid)

#Get the quote 
quote = web.DataReader(var, data_source = 'yahoo', start = '2012-01-01', end=now)

#create a new data frame
new_df = quote.filter(['Close'])

#get the last 60 day closing price values and convert the data frame to an array
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#create an empty list
X_test = []

#append the past 60 days
X_test.append(last_60_days_scaled)

#convert the X_test data set to a numpy array
X_test = np.array(X_test)

#reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#get the predicted scaled price
pred_price = model.predict(X_test)

#undo the scaling
pred_price = scaler.inverse_transform(pred_price)

quote2 = web.DataReader(var, data_source = 'yahoo', start = "2019-12-18", end = now)

print(quote2['Close'])

print('Predicted Price', pred_price)