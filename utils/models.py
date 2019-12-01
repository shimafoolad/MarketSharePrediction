from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class models():
    def __init__(self, data, mode):
        # Splitting the data for training and testing
        self.data = data
        features = self.data.drop('Market Share_total', 1)
        labels = self.data['Market Share_total']

        # feature scaling
        sc_X = StandardScaler()

        if mode == 'train':
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(features, labels, test_size=0.2, random_state=8)
            self.x_train = sc_X.fit_transform(self.x_train)
            self.x_val = sc_X.transform(self.x_val)
        elif mode == 'test':
            self.x_val = features
            self.y_val = labels
            self.x_val = sc_X.fit_transform(self.x_val)



    def getLRmodel(self):
        return self.model_LR

    def getRFmodel(self):
        return self.model_RF

    def Prediction(self, model):
        self.preds = model.predict(self.x_val)

    def LinearRegressionModel(self):
        self.model_LR = LinearRegression()
        self.model_LR.fit(self.x_train, self.y_train)

    def RandomForestModel(self):
        self.model_RF = RandomForestRegressor(n_estimators=20)
        self.model_RF.fit(self.x_train, self.y_train)

    def MAE_metric(self):
        return mean_absolute_error(y_true=self.y_val, y_pred=self.preds)

    def RSquared_metric(self):
        return r2_score(y_true=self.y_val, y_pred=self.preds)
