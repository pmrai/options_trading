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

quandl.ApiConfig.api_key = 'GuTeqGsJBkf1sn8DAa9G'
symbol = 'VIX'

# def get_quandl_data():
# 	# start_date = "2013-12-31"
# 	# end_date = "2018-08-30"
# 	df = quandl.get("CBOE/" + symbol, start_date = "2014-12-31", end_date = "2018-08-20" )
# 	df.to_pickle('%s_quandl.pk'%(symbol))
# 	df.to_csv('%s_quandl.csv'%(symbol))


def extract_symbol_from_data():
	path = [r'/Users/pmrai/research/options_trading/data/raw/2017November/*.csv']
	#path = [r'/Users/pmrai/research/options_trading/data/raw/2017December/*.csv',
	#r'/Users/pmrai/research/options_trading/data/raw/2017November/*.csv'
#] # use your path
	#symbol = 'AAPL'
	for direc in path:
		print(direc)
		symbol_df = pd.DataFrame()
		folder = iglob(direc,recursive=True) 
		for date_file in folder:
		    df = pd.read_csv(date_file)
		    df = df.loc[df['Symbol'] == symbol]
		    if df.empty is False:
		    	symbol_df = symbol_df.append(df)
	symbol_df.to_pickle('%s.pk'%(symbol))

def clean_options_data(df):
    df = df[['Symbol','ExpirationDate','LastPrice','PutCall','StrikePrice','Volume','ImpliedVolatility','Delta','Vega','UnderlyingPrice','DataDate']] #select only required columns
    df = df.loc[df['Volume'] > 10]
    df = df.loc[df['PutCall'] == 'call']
    df['DataDate'] = pd.to_datetime(df['DataDate'])
    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'])
    df['Days'] = [int(i.days) for i in (df.ExpirationDate - df.DataDate)]
    df = df.loc[df['Days'] > 0]
    df = df.reset_index(drop=True)
    today = pd.datetime.today()
    df = df.loc[df['ExpirationDate']<today]
    df.to_pickle('%s_clean.pk'%(symbol))

def asset_value_on_expiry(df):
	expiration_dates = df.ExpirationDate.unique()
	print(expiration_dates)
	qdl_pd = pd.read_pickle('%s_quandl.pk'%(symbol))
	print(qdl_pd)
	s = df.Symbol[0]
	df_with_expiry = pd.DataFrame()
	market_value_dict = {}
	for date in expiration_dates:
		date = pd.to_datetime(date)
		#market_value_on_date = quandl.get("WIKI/" + s, start_date = date, end_date = date)
		if date in qdl_pd.index:
			value = qdl_pd.loc[date]['VIX Close']	
			market_value_dict[date] = value
		else:
			print(date)
			df = df.drop(df[df.ExpirationDate == date].index)
	print(market_value_dict)
	df["ActualMarketValue"] = df["ExpirationDate"].map(market_value_dict)
	df_with_expiry = df
	df_with_expiry.to_pickle('%s_with_expiry.pk'%(symbol))

# get_quandl_data()
# quandl_df = pd.read_pickle('%s_quandl.pk'%(symbol))
# print(quandl_df)
# extract_symbol_from_data()
# symbol_df = pd.read_pickle('%s.pk'%(symbol))
# clean_options_data(symbol_df)
# clean_symbol_df = pd.read_pickle('%s_clean.pk'%(symbol))
# asset_value_on_expiry(clean_symbol_df)
# df = pd.read_pickle('%s_with_expiry.pk'%(symbol))
# df['P_L'] = [max(-x,y-(x+z)) for x,y,z in zip(df.LastPrice, df.ActualMarketValue, df.StrikePrice)]
# df = df.sort_values('Days')
# df.to_pickle('%s_with_pl.pk'%(symbol))
# df.to_csv('%s_with_pl.csv'%(symbol))
df = pd.read_pickle('%s_with_pl.pk'%(symbol))
#print(df.Days.unique())
df = df.loc[df['Days'] == 5]
#print(df)
df = df.reset_index(drop=True)
# fig = plt.figure()
# #plt.plot(df['P_L'])
# df['P_L'].hist(bins=10)
# plt.show()
# fig.savefig('day_5_nov_dec.png')
#print(df)

df_train=df.sample(frac=0.5,random_state=200)
df_test=df.drop(df_train.index)

X_train = df_train[['Delta','Vega']]
y_train = df_train[['P_L']]

X_test = df_test[['Delta','Vega']]
y_test = df_test[['P_L']]

lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
df_test['predictions'] = predictions
print(df_test)
# Model comparison
df_test_random = df_test.sample(frac=0.5, random_state=100)
df_test_predicted = df_test.nlargest(25, ['predictions'], keep='first')

#fig_comparison = plt.figure()
#plt.plot(df['P_L'])
# df_test_random['P_L'].hist(bins=4)
# df_test_predicted['P_L'].hist(bins=4)
# plt.show()
# fig_comparison.savefig('day_5_nov_dec.png')


#plt.show()

print(df_test_random)
print(df_test_predicted)
df_comparison = pd.DataFrame()

df_test_random =  df_test_random.reset_index(drop=True)
df_test_predicted = df_test_predicted.reset_index(drop=True)

df_comparison['random strategy'] = df_test_random['P_L']
df_comparison['risk managed strategy'] = df_test_predicted['P_L']
print(df_comparison)

fig1 = plt.figure()
df_comparison.plot.kde()
plt.show()
fig1.savefig('strategy_comp.png')




