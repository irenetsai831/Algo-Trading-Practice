# Creating an trading strategy by SVM
# Using as "trend & rsi" features; "change%--direction" as labels
# Using "ParameterGrid" for Optimization

import pandas as pd
import numpy as np 
import yfinance as yf
import datetime 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid


def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return pd.DataFrame(data)

def calculate_rsi(data, period):
    data['move'] = data['Close'] - data['Close'].shift(1)
    data['up'] = np.where(data['move']>0, data['move'], 0)
    data['down'] = np.where(data['move']<0, data['move'], 0)
    data['average gain'] = data['up'].rolling(period).mean()
    data['average loss'] = data['down'].abs().rolling(period).mean()
    return 100 - (100/(1+data['average gain']/data['average loss']))

def construct_signals(data, sma_period=60, rsi_period=14): 
    data['SMA'] = data['Close'].rolling(sma_period).mean()
    
    # construct 2 signals 
    data['trend'] = (data['Open'] - data['SMA'])*100
    data['rsi'] = calculate_rsi(data, rsi_period)/100
    # labels
    data['direction'] = np.where(data['Close'] - data['Open'], 1, -1)


if __name__ == '__main__': 
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2022, 1, 1)
    
    # EUR-USD 
    currency_data = download_data('EURUSD=X', start_date, end_date)
    construct_signals(currency_data)
    currency_data = currency_data.dropna()
    
    X = currency_data[['trend', 'rsi']]
    y = currency_data['direction']
    
    # data split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # model - we can find the right coefficients
    parameters = {'gamma': [10, 1, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000, 10000]}
    grid = list(ParameterGrid(parameters))
    
    best_accuracy = 0
    best_parameters = None 
    
    for p in grid: 
        svm = SVC(C=p['C'], gamma=p['gamma'])
        svm.fit(X_train, y_train)
        predictions = svm.predict(X_test)
        
        print(p)
        if accuracy_score(y_test, predictions) > best_accuracy: 
            best_accuracy = accuracy_score(y_test, predictions)
            best_parameters = p
        
 
    # we have found the best parameters
    model = SVC(C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print('------------------------------------------')
    print('Accuracy of the model: %.2f' % accuracy_score(y_test, predictions))
    print(p)
    print(confusion_matrix(predictions, y_test))
