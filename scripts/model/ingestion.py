from glob import iglob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import quandl
from scipy.stats import norm
from sklearn import linear_model
import os.path
from pathlib import Path

quandl.ApiConfig.api_key = 'GuTeqGsJBkf1sn8DAa9G'

def get_data_path():
	script_file = os.path.dirname(os.path.realpath('__file__'))
	current_dir = str(Path(script_file).parents[1])
	data_dir_path = os.path.join(current_dir,'data')
	return data_dir_path

def get_quandl_data(symbol):
	data_dir_path = get_data_path() 
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_quandl.pk'%(symbol))
	print(read_file)
	if(os.path.exists(read_file)):
		print('%s_quandl.pk exists in the preprocessed folder'%(symbol))
	else:
		df = quandl.get("WIKI/"+symbol, start_date = "2010-12-31", end_date = "2018-08-20")
		write_path = os.path.join(data_dir_path,'preprocessed')
		write_file = os.path.join(write_path,'%s_quandl.pk'%(symbol))
		df.to_pickle(write_file)

def SubDirPath (d):
    return list(filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)]))

def extract_symbol_from_data(symbol):
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s.pk exists in the preprocessed data folder'%(symbol))
	else:
		raw_data_dir_path = os.path.join(data_dir_path,'raw')
		symbol_df = pd.DataFrame()
		for ind in range(1,13):
			month_csv = r'2017_%s/*.csv'%(str(ind))
			dr = os.path.join(raw_data_dir_path,month_csv)
			print(dr)
			month_folder = iglob(dr,recursive=True)
			for date_file in month_folder:
			    df = pd.read_csv(date_file)
			    df = df.loc[df['Symbol'] == symbol]
			    if df.empty is False:
			    	symbol_df = symbol_df.append(df)
		write_path = os.path.join(data_dir_path,'preprocessed')
		write_file = os.path.join(write_path,'%s.pk'%(symbol))
		symbol_df.to_pickle(write_file)

def clean_options_data(symbol):
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_clean.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s_clean.pk exists in the preprocessed data folder'%(symbol))
	else:
	    read_file = os.path.join(read_path,'%s.pk'%(symbol))
	    df = pd.read_pickle(read_file)
	    df = df[['Symbol','ExpirationDate','LastPrice','PutCall','StrikePrice','Volume','ImpliedVolatility','Delta','Vega','UnderlyingPrice','DataDate']]
	    df = df.loc[df['Volume'] > 0]
	    df = df.loc[df['PutCall'] == 'call']
	    df['DataDate'] = pd.to_datetime(df['DataDate'])
	    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'])
	    df['Days'] = [int(i.days) for i in (df.ExpirationDate - df.DataDate)]
	    df = df.loc[df['Days'] > 0]
	    df = df.reset_index(drop=True)
	    today = pd.datetime.today()
	    df = df.loc[df['ExpirationDate']<today]
	    write_path = os.path.join(data_dir_path,'preprocessed')
	    write_file = os.path.join(write_path,'%s_clean.pk'%(symbol))
	    df.to_pickle(write_file)
    

def asset_value_on_expiry(symbol):
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_with_val_on_expiry.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s_with_val_on_expiry.pk exists in the preprocessed data folder'%(symbol))
	else:	
		read_file = os.path.join(read_path,'%s_clean.pk'%(symbol))
		df = pd.read_pickle(read_file)
		expiration_dates = df.ExpirationDate.unique()
		qdl_pd = pd.read_pickle(os.path.join(read_path,'%s_quandl.pk'%(symbol)))
		df_with_expiry = pd.DataFrame()
		market_value_dict = {}
		for date in expiration_dates:
			date = pd.to_datetime(date)
			if date in qdl_pd.index:
				value = qdl_pd.loc[date]['Close']	
				market_value_dict[date] = value
			else:
				df = df.drop(df[df.ExpirationDate == date].index)
		df["ActualMarketValue"] = df["ExpirationDate"].map(market_value_dict)
		write_path = os.path.join(data_dir_path,'preprocessed')
		write_file = os.path.join(write_path,'%s_with_val_on_expiry.pk'%(symbol))
		df_with_expiry = df
		df_with_expiry.to_pickle(write_file)

def get_profit_on_options(symbol):
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_with_profit.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s_with_profit.pk exists in the preprocessed data folder ...'%(symbol))
	else:
		read_file = os.path.join(read_path,'%s_with_val_on_expiry.pk'%(symbol))
		df = pd.read_pickle(read_file)
		df['P_L'] = [max(-x,y-(x+z)) for x,y,z in zip(df.LastPrice, df.ActualMarketValue, df.StrikePrice)]
		df = df.sort_values('Days')
		df = df[['DataDate','ExpirationDate','LastPrice','ImpliedVolatility','Delta','Vega','P_L','Days']]
		write_path = os.path.join(data_dir_path,'preprocessed')
		write_file = os.path.join(write_path,'%s_with_profit.pk'%(symbol))
		df.to_pickle(write_file)

def sort_profit_data_date(symbol):
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_date_sorted.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s_date_sorted.pk exists in the preprocessed data folder ...'%(symbol))
	else:
		read_file = os.path.join(read_path,'%s_with_profit.pk'%(symbol))
		df = pd.read_pickle(read_file)
		print(df)
		df = df.groupby('Days').apply(lambda x: x.sort_values(['DataDate']))
		df.reset_index(drop = True, inplace = True)
		df.to_pickle('pltest.csv')
		write_path = os.path.join(data_dir_path,'processed')
		write_file = os.path.join(write_path,'%s.pk'%(symbol))
		df.to_pickle(write_file)
		df.to_csv('test.csv')

def extract_data_for_model_building(symb):
	get_quandl_data(symb)
	extract_symbol_from_data(symb)
	clean_options_data(symb)
	asset_value_on_expiry(symb)
	get_profit_on_options(symb)
	sort_profit_data_date(symb)

def main():
	extract_data_for_model_building(symb)



if __name__ == '__main__':
	print('Creating data pipeline....')
	main()
	





