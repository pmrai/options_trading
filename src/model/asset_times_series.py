import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os.path
from pathlib import Path

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


symbol = 'GOOGL'
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


def read_asset_date():
    data_dir_path = get_data_path()
    read_path = os.path.join(data_dir_path,'preprocessed')    
    exchange_data = pd.read_pickle(os.path.join(read_path,'%s_quandl.pk'%(symbol)))
    exchange_data["Date"] = pd.to_datetime(exchange_data.index)
    ind_exchange_data = exchange_data.set_index(["Date"], drop=True)
    data_frame = ind_exchange_data.sort_index(axis=1,ascending=True)
    df = data_frame.to_pickle('%s_stock.pk'%(symbol))

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))

def train_test_split():
    data_dir_path = get_data_path()
    read_path = os.path.join(data_dir_path,'preprocessed')
    df = pd.read_pickle(os.path.join(read_path,'%s_quandl.pk'%(symbol)))
    df = df[['Close']]
    print(df.columns)

    split_date = pd.Timestamp('08-02-2017')

    train = df.loc[:split_date]
    test = df.loc[split_date:]

    test.to_csv('test_prediction_dates.csv')


    sc = MinMaxScaler()
    train_sc = sc.fit_transform(train)
    test_sc = sc.transform(test)

    X_train = train_sc[:-1]
    y_train = train_sc[1:]

    X_test = test_sc[:-1]
    y_test = test_sc[1:]


    return X_train, y_train, X_test, y_test

def fit_LSTM(X_train,y_train,X_test,y_test):
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
    model_lstm.save('LSTM_NonShift.h5')
    return X_tst_t

def prediction_LSTM(X_tst_t, y_test):
    model_lstm = load_model('LSTM_NonShift.h5')
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
    y_test = plot_df['True']
    y_pred_test_LSTM = plot_df['LSTM_prediction']
    plt.plot(y_test, label='True')
    plt.plot(y_pred_test_LSTM, label='LSTM')
    plt.title("LSTM's_Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Asset scaled')
    plt.legend()
    plt.show()

def gather_prediction_data():
    val_df = pd.read_csv('PredictionResults_LSTM_NonShift.csv')
    date_df = pd.read_csv('test_prediction_dates.csv')
    df = pd.DataFrame()
    df['Date'] = pd.to_datetime(date_df['Date'])
    df['LSTM_prediction'] = val_df['LSTM_prediction']
    df.to_csv('date_lstm_predict.csv')
    return df

def run_time_series_prediction():
    read_asset_date()
    (X_train,y_train,X_test,y_test) = train_test_split()
    X_tst_t = fit_LSTM(X_train,y_train,X_test,y_test)
    y_pred_test_LSTM = prediction_LSTM(X_tst_t, y_test)
    results = make_time_series_dataframe(y_test, y_pred_test_LSTM)
    plot_df = pd.read_csv('PredictionResults_LSTM_NonShift.csv')
    print(plot_df)
    plot_time_series_prediction(plot_df)

def get_expiry_dates():
    data_dir_path = get_data_path()
    read_path = os.path.join(data_dir_path,'preprocessed')
    read_file = os.path.join(read_path,'%s_with_profit.pk'%(symbol))
    df_all = pd.read_pickle(read_file)
    df_all['ExpirationDate'] = pd.to_datetime(df_all['ExpirationDate'])
    df_exp_dates = df_all.loc[df_all['ExpirationDate']>inv_date]
    return df_exp_dates

def get_expection_on_expiry_dates():
    df_test_dates = gather_prediction_data()
    df_test_dates = df_test_dates.loc[df_test_dates['LSTM_prediction']>0]
    df_exp_dates = get_expiry_dates()
    expiration_dates = df_exp_dates.ExpirationDate.unique()
    df_test_dates = df_test_dates.set_index(['Date'])
    df = pd.DataFrame({'Date':expiration_dates})
    df_test_dates =  df_test_dates[df_test_dates.index.isin(expiration_dates)]
    df_test_dates = df_test_dates[df_test_dates['LSTM_prediction'] >= 0.03]
    df_test_dates['Days'] = df_test_dates.index-inv_date
    print(df_test_dates)

  
    
def prediction():
    run_time_series_prediction()
    get_expection_on_expiry_dates()

prediction()



#get_expection_on_expiry_dates()
# gather_prediction_data()
# plot_df = pd.read_csv('PredictionResults_LSTM_NonShift.csv')
# plot_time_series_prediction(plot_df)
# run_time_series_prediction()