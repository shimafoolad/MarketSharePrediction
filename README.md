# MarketSharePrediction
Market share prediction using Linear Regression and Random Forest Regression

## Requirements

* numpy 1.17.4
* pandas 0.25.3
* scikit-learn 0.21.3

## How to run
- install requirements (pip install -r requirements.txt)
- download data.csv from https://drive.google.com/open?id=1JNyXs8CPbqneo99rGRJswIM7G1-Zsldp and put it into data folder
- download test.csv from https://drive.google.com/open?id=1C1eDzYSWy5vyhD--4Dn8tS_ev-wpUrOp and put it into data folder
- run main.py file.

## Results

We Split the data.csv for training and validation (0.8 - 0.2). 
results of validation have been shown as follows:


Models 			                | Val MAE  | Val R-Squared
----------------------------| :------: | :------------:
Linear Regression           | 0.018    | 0.627
Random Forest Regression    | 0.013    | 0.786
