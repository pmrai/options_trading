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
import matplotlib.animation as animation
import time
import argparse

#extract_data_for_model_building()
#symbol = 'GOOGL'
inv_date = pd.to_datetime('2017-08-01')

def regression_model(symbol):
	#extract_data_for_model_building()
	data_dir_path = get_data_path()
	read_path = os.path.join(data_dir_path,'processed')
	read_file = os.path.join(read_path,'%s.pk'%(symbol))
	df_all = pd.read_pickle(read_file)
	list_of_days = df_all.Days.unique()
	days = []
	median_Random = []
	low_Random = []
	high_Random = []
	median_Managed = []
	low_Managed = []
	high_Managed = []
	df_plot = pd.DataFrame()
	df_inv = pd.DataFrame()
	print(list_of_days)
	for day in list_of_days[30:60]:
		df = df_all.loc[df_all['Days'] == day]


		df_train = df[df['DataDate']<inv_date]
		df_test = df[df['DataDate']>inv_date] 

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

			df_test_predicted = df_test.nlargest(20,['predictions'], keep='first')

			df_comparison = pd.DataFrame()
			df_percent_profit = pd.DataFrame()

			df_test_random =  df_test_random.reset_index(drop=True)
			df_test_predicted = df_test_predicted.reset_index(drop=True)


			df_comparison['random strategy'] = df_test_random['P_L']
			df_comparison['risk managed strategy'] = df_test_predicted['P_L']

			print(day)
			 
			df_percent_profit['P_L_Percent_Managed'] =  (df_test_predicted['P_L']/df_test_predicted['LastPrice'])*100
			df_percent_profit['P_L_Percent_Random'] =  (df_test_random['P_L']/df_test_random['LastPrice'])*100
			df_percent_profit['Day'] = day

			df_inv = df_inv.append(df_percent_profit)

			print(df_percent_profit)

			days.append(day)
			median_Random.append(df_percent_profit['P_L_Percent_Random'].median())
			low_Random.append(df_percent_profit['P_L_Percent_Random'].quantile(q=0.25))
			high_Random.append(df_percent_profit['P_L_Percent_Random'].quantile(q=0.75))
			median_Managed.append(df_percent_profit['P_L_Percent_Managed'].median())
			low_Managed.append(df_percent_profit['P_L_Percent_Managed'].quantile(q=0.25))
			high_Managed.append(df_percent_profit['P_L_Percent_Managed'].quantile(q=0.75))

			df = {'Days':days,'median_Random':median_Random,'low_Random':low_Random,'high_Random':high_Random,\
					'median_Managed':median_Managed,'low_Managed':low_Managed,'high_Managed':high_Managed}
			df_plot = pd.DataFrame(df)
			df_plot = df_plot.set_index(['Days'])
			df_plot.to_csv('quantile_vals')
			print(df_plot)

	return df_plot, df_inv


def plot_results(df,symbol):
	font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
	strategies = ['Random','Managed']
	for i,strategy in enumerate(strategies):
		plt.figure(i+1)
		print(strategy)
		plt.errorbar(df.index, df['median_%s'%(strategy)], yerr=(df['median_%s'%(strategy)]-df['low_%s'%(strategy)],df['high_%s'%(strategy)]-df['median_%s'%(strategy)]),fmt='o', color='blue',
	        ecolor='lightgray', elinewidth=3, capsize=0)
		plt.title('(%s Strategy)'%(strategy), fontdict=font)
		plt.xlabel('Investment Duration (Days)', fontdict=font)
		plt.ylabel('Expected Return (Percentage)', fontdict=font)
		plt.ylim(-300,300)
	plt.show()	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("symb", help ="SYMBOL for the underlying stock " + \
	"you wish to trade options on", type = str)
	args = parser.parse_args()
	extract_data_for_model_building(args.symb)
	df, df_inv = regression_model(args.symb)
	#strategy = 'Random'
	plot_results(df,args.symb)

if __name__ == '__main__':
	print('Classification using regression model....')
	main()
#regression_model()