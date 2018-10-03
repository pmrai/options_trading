import sys
sys.path.insert(0, './../../src/ingestion')
import argparse
import pandas as pd
from ingestion import extract_data_for_model_building
from regression_model import regression_model, plot_results
from asset_times_series import run_time_series_prediction, get_optimal_expiry_dates
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symb", help ="SYMBOL for the underlying stock " + \
                    "you wish to trade options on", type = str)
    args = parser.parse_args()
    extract_data_for_model_building(args.symb)
    df, df_inv = regression_model(args.symb)
    opt_days = get_optimal_expiry_dates(args.symb)
    get_investment_gain(df_inv, opt_days)

def get_investment_gain(df_inv, opt_days):
    random_profit = []
    managed_profit = []
    print(df_inv)
    inv_days = []
    random_strategy = []
    managed_strategy = []
    for day in opt_days:
        df_select = df_inv.loc[df_inv['Day'] == day]
        if df_select.empty:
            pass
        else: 
            inv_days.append(day)
            random_strategy.append(df_select['P_L_Percent_Random'].sample(1, random_state=100))
            managed_strategy.append(df_select['P_L_Percent_Managed'].iloc[0])
    df = {'Invest_Days':inv_days,'Random':random_strategy,'Managed':managed_strategy}
    df_opt_comp = pd.DataFrame(df)
    df_opt_comp = df_opt_comp.set_index(['Invest_Days'])
    print("Random Strategy Mean is: %d", df_opt_comp['Random'].mean())
    print("Managed Strategy Mean is: %d", df_opt_comp['Managed'].mean())
    print("Random Strategy Std is: %d", df_opt_comp['Random'].std())
    print("Managed Strategy Std is: %d", df_opt_comp['Managed'].std())



if __name__ == '__main__':
    print('Testing options trading....')
    main()


