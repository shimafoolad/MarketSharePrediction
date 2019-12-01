import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime as dt


class preprocess():
    def __init__(self, data):
        # remove less important features
        # including 'Unnamed: 0', 'Episode', 'Name of show', 'Name of episode' and 'Date'
        self.data = data.drop(['Unnamed: 0', 'Episode', 'Name of show', 'Name of episode', 'Date'], 1)

        # Encode the Categorical data
        self.EncodeCategoricalData()

        # convert Market Share_total feature from [0, 100] to range [0,1]
        market_share = np.array(self.data['Market Share_total'])
        self.data['Market Share_total'] = market_share / 100

        # check out the missing values
        self.data[self.data == np.inf] = np.nan
        self.data.fillna(self.data.mean(), inplace=True)

    ''' return data
    '''
    def getData(self):
        return self.data

    ''' Encode the Categorical data
    '''
    def EncodeCategoricalData(self):
        # convert Start_time and End_time features (columns) to numerical.
        # our ML model (Linear regression and random forest regression) doesn't work on date data.
        # Therefore we need to convert it into numerical value.
        date_features = ['Start_time', 'End_time']
        self.ConvertDateTimeToNumerical(date_features)

        # Encode the Categorical data
        # convert categorical features with Yes and No values to numerical
        # (e.g. 'First time or rerun',
        #       '# of episode in the season',
        #       'Movie?',
        #       'Game of the Canadiens during episode?' features
        self.data = self.data.replace({'No': 0, 'Yes': 1})

        label_encoder = LabelEncoder()

        # encode 'Day of week' feature
        self.data['Day of week'] = self.data['Day of week'].map(
            {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4,
             'Saturday': 5, 'Sunday': 6})
        label_encoder.fit(self.data['Day of week'])
        label_encoder.transform(self.data['Day of week'])

        # encode 'Channel Type' feature
        self.data['Channel Type'] = self.data['Channel Type'].map({'General Channel': 0, 'Specialty Channel': 1})
        label_encoder.fit(self.data['Channel Type'])
        label_encoder.transform(self.data['Channel Type'])

        # apply one-hot encoding on nominal features (no numerical relationship between the different categories)
        # (e.g. 'Station',
        #       'Genre',
        #       'Season' features
        self.data = pd.get_dummies(self.data, prefix_sep='_', drop_first=True)


    def ConvertDateTimeToNumerical(self, features):
        for feat in features:
            self.data[feat] = pd.to_datetime(self.data[feat])
            self.data[feat] = self.data[feat].map(dt.datetime.toordinal)
