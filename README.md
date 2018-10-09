# Options Trading

This is my Insight Artificial Intelligence project. 

## What are Options

Options are trading instruments that are traded on top of an underlying asset (e.g. stock of a company). That is why they are called as derivatives. Options (specifically 'call' options) provide the the possibility of bounded risk with unbounded profit.


Let us understand this using an analogy. Let us say we have our childhood friends Mickey and Donald. Mickey lives in City A which is connected with San Francisco using BART. Donald owns a house in city B which is not connected with BART. There is a rumour that BART will be extended to city B and hence the price of Donald's price will shoot up. Mickey wants to to take advantage of this opportunity and offers a small premium upfront to lock the price of the house six months from now. In this deal, there are two future possibilities. If BART comes, Mickey will make profit by buying the house at a lower price than market price. If BART does not come, Mickey will loose the premium offered. So Mickey's loss is bounded but profit is unbounded. Higher the increase in the price of the house, greater the profit. In this analogy, we can replace the house with an underlying stock asset.

Options trading is an AI platform that suggests risk bounded options selection. It uses a combination of neural network based Time series model and simple regression based classifier to select the best options portfolio.

For time series model, we use stock price data of the undelying asset and for the classification model we use historical options chain data.

Google slides can be found [here](https://docs.google.com/presentation/d/1bubSDpVukkwkACXivLEY4nQMSzOSIPdzUTxachdkKQ4/edit#slide=id.g4302bf8b27_0_0)

## Data

The stock price data for time series modeling is obtained using quandl api. Historical options price data is not freely available. For this project, option chain data for the year 2017 is used.

Training time and investment duration:

For time series model, we use stock price data between from January 2010 to July 2017 for training. For classification model, we use option chain data between Jan 2017 to July 2017. We consider investment in options in the period between August 2017 to December 2017 for all expiry dates in this duration. 


## Model

## Time Series Model

Reinforcement learning is used for time series modeling. Specifically, a neural network with five LSTM nodes and one dense layer is used for time series prediction.There are two reasons for using LSTM. First, LSTM models do not require any assumption on stationarity of the time series. Second, LSTMs can be used for predicting several time stamps ahead in time while taking into account both long and short term trends.  Note that, since only call options are considered, only a general trend is required and not the exact stock value at a particular date in future. All functions related to time series prediction is coded in asset_times_series.py file

## Classification Model

The data pipeline for classification is coded in ingestion.py file. It reads historical option chain data, cleans it, gets the value of asset on expiry, determines profit/loss and makes the data ready for training. The classification model is coded in regression_model.py. In this model, the inputs are the so called greeks for a particular option and the output is a profit or loss label.  With this classification model, we can determine if a given option deal with specififed values of greeks will end up in profit or loss in future. 

## How to run the package

The ML model uses a simple command line interface as follows:

python options_trading <SYMBOL>

where SYMBOL is the undelying stock on which the option is traded (e.g. GOOGL for Google, VIX for Volatility Index etc.)
