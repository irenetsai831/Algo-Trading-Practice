# Creating an trading strategy by Logistic Regression 
# Using as "2 lags of data points" features; "change%--direction" as labels

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def download_data(stock, start, end): 
    data = {}
    ticker = yf.download(stock, start, end)
    data['Close'] = ticker['Adj Close']
    return pd.DataFrame(data)

def construct_features(data, lags=2): 
    
    # calculate the lagged adjusted closing prices
    for i in range(0, lags): 
        data['Lag%s' % str(i+1)] = data['Close'].shift(i+1)
    
    # calculate the percent of autual changes 
    data['Today Change'] = data['Close'].pct_change() * 100
    
    # calculate the lags in percentage (normalization)
    for i in range(0, lags): 
        data['Lag%s'%str(i+1)] = data['Lag%s'%str(i+1)].pct_change() * 100
        
    # direction -- the target variable 
    data['Direction'] = np.where(data['Today Change']>0, 1, -1)
    
    print(data)


if __name__ == '__main__': 
    start_date = datetime.datetime(2017, 1, 1)
    end_date = datetime.datetime(2022, 1, 1)
    
    stock_data = download_data('MSFT', start_date, end_date)
    construct_features(stock_data)
    stock_data = stock_data.dropna()
    
    # features and the labels (target variables)
    X = stock_data[['Lag1', 'Lag2']]
    y = stock_data['Direction']
    print(X)
    
    # split the data into training and testing dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # training the model on the training dataset 
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # test the model on the testing dataset 
    predictions = model.predict(X_test)
    
    print('Accuracy of the model: %.2f'%accuracy_score(y_test, predictions))
    print(confusion_matrix(predictions, y_test))
