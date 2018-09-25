from glob import iglob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from bs4 import BeautifulSoup
#import pandas_datareader.data as web
import random
import matplotlib.pyplot as plt
import quandl
from scipy.stats import norm
from sklearn import linear_model
import os.path
from pathlib import Path

quandl.ApiConfig.api_key = 'GuTeqGsJBkf1sn8DAa9G'
symbol = 'VIX'

def get_data_path():
	script_file = os.path.dirname(os.path.realpath('__file__'))
	current_dir = str(Path(script_file).parents[1])
	data_dir_path = os.path.join(current_dir,'data')
	return data_dir_path

def get_quandl_data():
	data_dir_path = get_data_path() 
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_quandl.pk'%(symbol))
	print(read_file)
	if(os.path.exists(read_file)):
		print('%s_quandl.pk exists in the preprocessed folder'%(symbol))
	else:
		df = quandl.get("CBOE/"+symbol, start_date = "2014-12-31", end_date = "2018-08-20")
		write_path = os.path.join(data_dir_path,'preprocessed')
		write_file = os.path.join(write_path,'%s_quandl.pk'%(symbol))
		df.to_pickle(write_file)

def SubDirPath (d):
    return list(filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)]))

def extract_symbol_from_data():
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s.pk exists in the preprocessed data folder'%(symbol))
	else:
		raw_data_dir_path = os.path.join(data_dir_path,'raw')
		symbol_df = pd.DataFrame()
		for ind in range(12,13):
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
	#symbol_df.to_csv('%s.csv'%(symbol))

def clean_options_data():
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
	    #df.to_pickle('%s_clean.pk'%(symbol))
	    #df.to_csv('%s_clean.csv'%(symbol))
    

def asset_value_on_expiry():
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_with_val_on_expiry.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s_with_val_on_expiry.pk exists in the preprocessed data folder'%(symbol))
	else:	
		read_file = os.path.join(read_path,'%s_clean.pk'%(symbol))
		df = pd.read_pickle(read_file)
		expiration_dates = df.ExpirationDate.unique()
		qdl_pd = pd.read_pickle('%s_quandl.pk'%(symbol))
		df_with_expiry = pd.DataFrame()
		market_value_dict = {}
		for date in expiration_dates:
			date = pd.to_datetime(date)
			if date in qdl_pd.index:
				value = qdl_pd.loc[date]['VIX Close']	
				market_value_dict[date] = value
			else:
				df = df.drop(df[df.ExpirationDate == date].index)
		df["ActualMarketValue"] = df["ExpirationDate"].map(market_value_dict)
		write_path = os.path.join(data_dir_path,'preprocessed')
		write_file = os.path.join(write_path,'%s_with_val_on_expiry.pk'%(symbol))
		df_with_expiry = df
		df_with_expiry.to_pickle(write_file)

def get_profit_on_options():
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
		df = df[['DataDate','ExpirationDate','ImpliedVolatility','Delta','Vega','P_L','Days']]
		write_path = os.path.join(data_dir_path,'preprocessed')
		write_file = os.path.join(write_path,'%s_with_profit.pk'%(symbol))
		df.to_pickle(write_file)

def sort_profit_data_date():
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'preprocessed')
	read_file = os.path.join(read_path,'%s_date_sorted.pk'%(symbol))
	if(os.path.exists(read_file)):
		print('%s_date_sorted.pk exists in the preprocessed data folder ...'%(symbol))
	else:
		read_file = os.path.join(read_path,'%s_with_profit.pk'%(symbol))
		df = pd.read_pickle(read_file)
		#df = df.set_index(['Days'], inplace=True)
		#df = df.reset_index(drop = True)
		print(df)
		df = df.groupby('Days').apply(lambda x: x.sort_values(['DataDate']))
		df.reset_index(drop = True, inplace = True)
		df.to_pickle('pltest.csv')
		write_path = os.path.join(data_dir_path,'processed')
		write_file = os.path.join(write_path,'%s.pk'%(symbol))
		df.to_pickle(write_file)
		df.to_csv('test.csv')

def extract_data_for_model_building():
	get_quandl_data()
	# quandl_df = pd.read_pickle('%s_quandl.pk'%(symbol))
	# print(quandl_df)
	extract_symbol_from_data()
	# symbol_df = pd.read_pickle('%s.pk'%(symbol))
	clean_options_data()
	# clean_symbol_df = pd.read_pickle('%s_clean.pk'%(symbol))
	asset_value_on_expiry()
	get_profit_on_options()
	sort_profit_data_date()
	# df = pd.read_pickle('%s_with_expiry.pk'%(symbol))
	# df['P_L'] = [max(-x,y-(x+z)) for x,y,z in zip(df.LastPrice, df.ActualMarketValue, df.StrikePrice)]
	# df = df.sort_values('Days')
	# df.to_pickle('%s_with_pl.pk'%(symbol))
	# df.to_csv('%s_with_pl.csv'%(symbol))
	# df = pd.read_pickle('%s_with_pl.pk'%(symbol))

extract_data_for_model_building()	

# df = df.loc[df['Days'] == 1]

# df = df.reset_index(drop=True)


# df_train=df.sample(frac=0.5,random_state=200)
# df_test=df.drop(df_train.index)

# X_train = df_train[['Delta','Vega']]
# y_train = df_train[['P_L']]

# X_test = df_test[['Delta','Vega']]
# y_test = df_test[['P_L']]

# lm = linear_model.LinearRegression()
# model = lm.fit(X_train,y_train)
# predictions = lm.predict(X_test)
# df_test['predictions'] = predictions
# print(df_train)
# print(df_test)
# # Model comparison
# df_test_random = df_test.sample(frac=0.2, random_state=100)
# df_test_predicted = df_test.nlargest(20, ['predictions'], keep='first')


# print(df_test_random)
# print(df_test_predicted)
# df_comparison = pd.DataFrame()

# df_test_random =  df_test_random.reset_index(drop=True)
# df_test_predicted = df_test_predicted.reset_index(drop=True)

# df_comparison['random strategy'] = df_test_random['P_L']
# df_comparison['risk managed strategy'] = df_test_predicted['P_L']
# print(df_comparison)

# fig1 = plt.figure()
# df_comparison.plot.kde()
# plt.show()
# fig1.savefig('strategy_comp.png')




