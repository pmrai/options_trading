import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os.path
from pathlib import Path
import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
import datetime


#symbol = 'GOOGL'
inv_date = datetime.datetime(2017, 8, 1, 0, 0)

def get_data_path():
    script_file = os.path.dirname(os.path.realpath('__file__'))
    current_dir = str(Path(script_file).parents[1])
    data_dir_path = os.path.join(current_dir,'data')
    return data_dir_path

def write_to_preprocessed(df, filename):
    write_path = os.path.join(get_data_path(),'preprocessed')
    write_file = os.path.join(write_path,filename)
    df.to_csv(write_file)


def read_asset_date(symbol):
    data_dir_path = get_data_path()
    read_path = os.path.join(data_dir_path,'preprocessed')    
    exchange_data = pd.read_pickle(os.path.join(read_path,'%s_quandl.pk'%(symbol)))
    #val_on_inv_date = exchange_data.loc[pd.Timestamp('08-01-2017')]['Close']
    exchange_data["Date"] = pd.to_datetime(exchange_data.index)
    ind_exchange_data = exchange_data.set_index(["Date"], drop=True)
    df = ind_exchange_data.sort_index(axis=1,ascending=True)
    test = df.loc[pd.Timestamp('08-01-2017'):]
    sc = MinMaxScaler()
    test_sc = sc.transform(test)
    val_inv_date_sc = test_sc.loc[pd.Timestamp('08-01-2017')]['Close']
    print(val_inv_date_sc)
    #df.to_pickle('%s_stock.pk'%(symbol))

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))

def train_test_split(symbol):
    data_dir_path = get_data_path()
    read_path = os.path.join(data_dir_path,'preprocessed')
    df = pd.read_pickle(os.path.join(read_path,'%s_quandl.pk'%(symbol)))
    df = df[['Close']]
    split_date = pd.Timestamp('08-01-2017')
    train = df.loc[:split_date]
    test = df.loc[split_date:]
    test.to_csv('test_prediction_dates.csv')
    sc = MinMaxScaler()
    train_sc = sc.fit_transform(train)
    test_sc = sc.transform(test)
    inv_val_sc = test_sc[0]
    X_train = train_sc[:-1]
    y_train = train_sc[1:]
    X_test = test_sc[:-1]
    y_test = test_sc[1:]
    return X_train, y_train, X_test, y_test, inv_val_sc

def fit_LSTM(X_train,y_train,X_test,y_test,symbol):

    X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    model_lstm = Sequential()
    model_lstm.add(LSTM(4, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])

    y_pred_test_lstm = model_lstm.predict(X_tst_t)
    y_train_pred_lstm = model_lstm.predict(X_tr_t)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
    r2_train = r2_score(y_train, y_train_pred_lstm)
    print("The Adjusted R2 score on the Train set is:\t{:0.3f}\n".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
    r2_test = r2_score(y_test, y_pred_test_lstm)
    print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1]))) 
    model_lstm.save('LSTM_NonShift_%s.h5'%(symbol))

def prediction_LSTM(X_tst_t, y_test,symbol):
    model_lstm = load_model('LSTM_NonShift_%s.h5'%(symbol))
    score_lstm= model_lstm.evaluate(X_tst_t, y_test, batch_size=1)
    print('LSTM: %f'%score_lstm)
    y_pred_test_LSTM = model_lstm.predict(X_tst_t)
    return y_pred_test_LSTM

def make_time_series_dataframe(y_test, y_pred_test_LSTM):
    col1 = pd.DataFrame(y_test, columns=['True'])
    col2 = pd.DataFrame(y_pred_test_LSTM, columns=['LSTM_prediction'])
    results = pd.concat([col1,col2], axis=1)
    results.to_csv('PredictionResults_LSTM_NonShift.csv')
    return results

def plot_time_series_prediction(plot_df):
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    y_test = plot_df['True']
    y_pred_test_LSTM = plot_df['LSTM_prediction']
    plt.plot(y_test, label='True')
    plt.plot(y_pred_test_LSTM, label='LSTM')
    plt.title('Google Stock Prediction using LSTM', fontdict=font)
    plt.xlabel('Investment Duration', fontdict=font)
    plt.ylabel('Normalized Stock Price', fontdict=font)
    # plt.title("LSTM's_Prediction")
    # plt.xlabel('Observation')
    # plt.ylabel('Asset scaled')
    plt.legend()
    #plt.show()

def gather_prediction_data():
    val_df = pd.read_csv('PredictionResults_LSTM_NonShift.csv')
    date_df = pd.read_csv('test_prediction_dates.csv')
    df = pd.DataFrame()
    df['Date'] = pd.to_datetime(date_df['Date'])
    df['LSTM_prediction'] = val_df['LSTM_prediction']
    df.to_csv('date_lstm_predict.csv')
    return df

def run_time_series_prediction(symbol):
    #read_asset_date(symbol)
    (X_train,y_train,X_test,y_test,inv_val) = train_test_split(symbol)
    lstm_model = 'LSTM_NonShift_%s.h5'%(symbol)
    if os.path.exists(lstm_model):
        pass
    else:
        fit_LSTM(X_train,y_train,X_test,y_test,symbol)
    X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_pred_test_LSTM = prediction_LSTM(X_tst_t,y_test,symbol)
    results = make_time_series_dataframe(y_test, y_pred_test_LSTM)
    plot_df = pd.read_csv('PredictionResults_LSTM_NonShift.csv')
    print(plot_df)
    plot_time_series_prediction(plot_df)
    return inv_val

def get_expiry_dates(symbol):
    data_dir_path = get_data_path()
    read_path = os.path.join(data_dir_path,'preprocessed')
    read_file = os.path.join(read_path,'%s_with_profit.pk'%(symbol))
    df_all = pd.read_pickle(read_file)
    df_all['ExpirationDate'] = pd.to_datetime(df_all['ExpirationDate'])
    df_exp_dates = df_all.loc[df_all['ExpirationDate']>inv_date]
    return df_exp_dates

def get_optimal_expiry_dates(symbol,inv_val):
    df_test_dates = gather_prediction_data()
    print(df_test_dates['LSTM_prediction'])
    df_test_dates = df_test_dates.loc[(df_test_dates['LSTM_prediction']-inv_val)>0]
    df_exp_dates = get_expiry_dates(symbol)
    expiration_dates = df_exp_dates.ExpirationDate.unique()
    df_test_dates = df_test_dates.set_index(['Date'])
    df = pd.DataFrame({'Date':expiration_dates})
    df_test_dates =  df_test_dates[df_test_dates.index.isin(expiration_dates)]
    df_test_dates = df_test_dates[df_test_dates['LSTM_prediction'] >= 0.03]
    df_test_dates['Days'] = df_test_dates.index-inv_date
    print(df_test_dates['Days'].dtype)
    print(df_test_dates['Days'].dt.days)
    opt_exp = df_test_dates.Days.dt.days
    return(opt_exp)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symb", help ="SYMBOL for the underlying stock " + \
                    "you wish to trade options on", type = str)
    args = parser.parse_args()
    inv_val = run_time_series_prediction(args.symb)
    get_optimal_expiry_dates(args.symb,inv_val)

if __name__ == '__main__':
    print('Running time series....')
    main()
