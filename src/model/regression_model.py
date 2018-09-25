import sys
sys.path.insert(0, './../../src/ingestion')
from ingestion import extract_data_for_model_building, get_data_path
from glob import iglob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import quandl
from scipy.stats import norm
from sklearn import linear_model
import os.path
from pathlib import Path

#extract_data_for_model_building()
symbol = 'VIX'

def regression_model():
	extract_data_for_model_building()
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'processed')
	read_file = os.path.join(read_path,'%s.pk'%(symbol))
	df_all = pd.read_pickle(read_file)
	list_of_days = df_all.Days.unique()
	print(list_of_days)
	for day in list_of_days[1:20]:
		#df = pd.DataFrame()
		df = df_all.loc[df_all['Days'] == day]


		#df = df.reset_index(drop=True)

		#df = df.set_index('DataDate')
		inv_date = pd.to_datetime('2017-03-01')
		df_train = df[df['DataDate']<inv_date]
		df_test = df[df['DataDate']>inv_date] 
		#print(df[df['DataDate']>inv_date])
		#df_train = df.sample(frac=0.5,random_state=200)
		#df_test = df.drop(df_train.index)
		if df_train.empty is False  and df_test.empty is False:  
			X_train = df_train[['Delta','Vega']]
			y_train = df_train[['P_L']]

			X_test = df_test[['Delta','Vega']]
			y_test = df_test[['P_L']]

			lm = linear_model.LinearRegression()
			model = lm.fit(X_train,y_train)
			predictions = lm.predict(X_test)
			df_test['predictions'] = predictions
			
			# Model comparison
			df_test_random = df_test.sample(20, random_state=100)
			#df_test_maxvol = df_test.nlargest(20,['Volume'], keep='first')
			#print(df_test_maxvol)
			df_test_predicted = df_test.nlargest(20,['predictions'], keep='first')
			df_test_vol = df_test.nlargest(20,['Volume'], keep='first')
			print(df_test_vol[['P_L','Volume','predictions']])

			# print(df_test_random)
			# print(df_test_predicted)
			df_comparison = pd.DataFrame()

			df_test_random =  df_test_random.reset_index(drop=True)
			df_test_predicted = df_test_predicted.reset_index(drop=True)
			df_test_volume = df_test_vol.reset_index(drop=True)

			df_comparison['random strategy'] = df_test_random['P_L']
			df_comparison['risk managed strategy'] = df_test_predicted['P_L']
			df_comparison['max vol strategy'] = df_test_volume['P_L']
			#df_comparison['max vol trategy'] = df_test_maxvol['P_L']
			print(df_comparison)

			#fig1 = plt.figure()
			df_comparison.plot.kde()
			plt.pause(0.05)
			plt.draw()
	plt.show()	
			#fig1.savefig('strategy_comp.png')

regression_model()