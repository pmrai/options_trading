import sys
sys.path.insert(0, './../../src/ingestion')
import argparse
import pandas as pd
from ingestion import extract_data_for_model_building
from regression_model import regression_model, plot_results
from asset_times_series import run_time_series_prediction, get_optimal_expiry_dates
from random import randint, seed

# import sys
# import os.path
# from pathlib import Path
# import argparse
# from glob import iglob

# import pandas as pd
# import numpy as np
# import quandl
# import random
# from random import randint, seed
# from scipy.stats import norm
# from sklearn import linear_model
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import r2_score
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# import keras.backend as K
# from keras.callbacks import EarlyStopping
# from keras.optimizers import Adam
# from keras.models import load_model
# from keras.layers import LSTM
# import time
# import datetime

# sys.path.insert(0, './../../scripts/ingestion')
# from ingestion import extract_data_for_model_building, get_data_path
# from regression_model import regression_model, plot_results
# from asset_times_series import run_time_series_prediction, get_optimal_expiry_dates

# import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symb", help ="SYMBOL for the underlying stock " + \
                    "you wish to trade options on", type = str)
    args = parser.parse_args()
    extract_data_for_model_building(args.symb)
    df, df_inv = regression_model(args.symb)
    inv_val = run_time_series_prediction(args.symb)
    opt_days = get_optimal_expiry_dates(args.symb,inv_val)
    get_investment_gain(df_inv,opt_days)

def get_investment_gain(df_inv, opt_days):
    random_profit = []
    managed_profit = []
    print(df_inv)
    inv_days = []
    random_strategy = []
    managed_strategy = []
    seed(100)
    for day in opt_days[1:10]:
        df_select = df_inv.loc[df_inv['Day'] == day]
        if df_select.empty:
            pass
        else: 
            inv_days.append(day)
            random_strategy.append(df_select['P_L_Percent_Random'].iloc[randint(0,10)])
            managed_strategy.append(df_select['P_L_Percent_Managed'].iloc[0])
    df = {'Invest_Days':inv_days,'Random':random_strategy,'Managed':managed_strategy}
    df_opt_comp = pd.DataFrame(df)
    df_opt_comp = df_opt_comp.set_index(['Invest_Days'])
    print("Random Strategy Median, 0.25 and 0.75 quantile is: ", df_opt_comp['Random'].median(),\
         df_opt_comp['Random'].quantile(q=0.25), df_opt_comp['Random'].quantile(q=0.75))
    print("Managed Strategy Median, 0.25 and 0.75 quantile is: ", df_opt_comp['Managed'].median(),\
        df_opt_comp['Managed'].quantile(q=0.25), df_opt_comp['Managed'].quantile(q=0.75))




if __name__ == '__main__':
    print('Testing options trading....')
    main()


