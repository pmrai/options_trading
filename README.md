
# Options Trading

![Alt text](./images/results.png?raw=true "Title")

This is my Insight Artificial Intelligence project. The main result is shown in the figure above. Here, I compare two different strategies to select options traded over Google stock. The first strategy (left) is selecting options at random. The second strategy (right) is machine learning informed option selection developed in this project. The blue dots in figures represent the median return of 20 options selected for a particular investment duration. The grey lines represent the variance of these returns. An options selection strategy is more predictable if the variance in returns is small. We clearly see that a machine learning strategy is more predictable with smaller spread around the median value.

For an easy and rspid introduction to options, see section 'what are options?' below.

Google slides can be found [here](https://docs.google.com/presentation/d/1bubSDpVukkwkACXivLEY4nQMSzOSIPdzUTxachdkKQ4/edit#slide=id.g4302bf8b27_0_0)

## How to run the package

The ML model uses a simple command line interface as follows:

 ```
python scripts/model/options_trading <SYMBOL>

```
where SYMBOL is the undelying stock on which the option is traded (e.g. GOOGL for Google etc.)

## Description

Options trading is an AI platform that suggests risk bounded options selection. It uses a combination of neural network based time series model (LSTM) and simple regression based classifier to select risk averse options portfolio.

For time series model, we use stock price data of the underlying asset and for the classification model we use historical options chain data.

# Data

The stock price data for time series modeling is obtained using quandl api. Historical options price data is not freely available and can be bought [here](http://www.cboe.com/data/historical-options-data). For this project, option chain data for the year 2017 is used.

![Alt text](./images/option_chain.png?raw=true "Title")

# Training time and investment duration:

For time series model, we use stock price data between from January 2010 to July 2017 for training. For classification model, we use option chain data between Jan 2017 to July 2017. We consider investment in options in the period between August 2017 to December 2017 for all expiry dates in this duration. 


# Model

# Time Series Model

Reinforcement learning is used for time series modeling. Specifically, a neural network with five LSTM nodes and one dense layer is used for time series prediction.There are two reasons for using LSTM. First, LSTM models do not require any assumption on stationarity of the time series. Second, LSTMs can be used for predicting several time stamps ahead in time while taking into account both long and short term trends.  Note that, since only call options are considered, only a general trend is required and not the exact stock value at a particular date in future. All functions related to time series prediction is coded in asset_time_series.py file.

![Alt text](./images/google_bull.png?=1x1)

# Classification Model

The data pipeline for classification is coded in ingestion.py file. It reads historical option chain data, cleans it, gets the value of asset on expiry, determines profit/loss and makes the data ready for training. The classification model is coded in regression_model.py. In this model, the inputs are the so called greeks for a particular option and the output is a profit or loss label.  With this classification model, we can determine if a given option deal with specififed values of greeks will end up in profit or loss in future. 

## What are Options?

Options are trading instruments that are traded on top of an underlying asset (e.g. stock of a company). That is why they are called as derivatives. Options (specifically 'call' options) provide the the possibility of bounded risk with unbounded profit.


Let us understand this using an analogy. Let us say we have our childhood friends Mickey and Donald. Mickey lives in City A which is connected with San Francisco using BART. Donald owns a house in city B which is not connected with BART. There is a rumour that BART will be extended to city B and hence the price of Donald's price will shoot up. Mickey wants to to take advantage of this opportunity and offers a small premium upfront to lock the price of the house six months from now. In this deal, there are two future possibilities. If BART is extended to city B, Mickey will make profit by buying the house at a lower price than market price. If BART is not extended, Mickey will loose the premium offered. So Mickey's loss is bounded but profit is unbounded. Higher the increase in the price of the house, greater the profit. In this analogy, we can replace the house with an underlying stock asset for options. Options price is the amount 

