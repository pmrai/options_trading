## Week 1: TO DO:

- Data preprocessing for corresponding Symbol, Date
- Data ingestion for corresponding input/output map
- Apply regression based classification
- Test if it works better than random picking of options

## Week 2:

- Added ingestion.py to create the data pipeline. It reads historical option chain data, cleans it, gets the value of asset on expiry, determines profit/loss and makes the data ready for training

- Added regression_model.py to build a regression based classification model for several expiry durations in days. The model reads delta and vega of an option and spits out estimated profit/loss. For a given duration, we choose 20 options which maximize predicted profit 