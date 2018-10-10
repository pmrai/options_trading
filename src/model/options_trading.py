import sys
sys.path.insert(0, './../../src/ingestion')
import argparse
import pandas as pd
from ingestion import extract_data_for_model_building
from regression_model import regression_model, plot_results
from asset_times_series import run_time_series_prediction, get_optimal_expiry_dates
from random import randint, seed

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
    print("Random Strategy Median is: %f,%f,%f", df_opt_comp['Random'].median(),\
         df_opt_comp['Random'].quantile(q=0.25), df_opt_comp['Random'].quantile(q=0.75))
    print("Managed Strategy Median is: %f,%f,%f", df_opt_comp['Managed'].median(),\
        df_opt_comp['Managed'].quantile(q=0.25), df_opt_comp['Managed'].quantile(q=0.75))




if __name__ == '__main__':
    print('Testing options trading....')
    main()


