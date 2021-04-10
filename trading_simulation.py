
import pandas as pd
import model_training as mdl
import datetime


def find_dates(data_y):
    data = data_y.copy()
    data['year'] = data['prediction_date'].dt.year
    data['month'] = data['prediction_date'].dt.month
    hyperparameter_retune_dates = data.groupby('year')['prediction_date'].min()
    model_retrain_dates = data.groupby(['year', 'month'])['prediction_date'].min()

    return hyperparameter_retune_dates, model_retrain_dates


class Portfolio:
    # write a portfolio DataFrame, whenever we enter a trade, add a row here
    def __init__(self):
        self.port_holdings = pd.DataFrame(columns={'trade_id',
                                                   'GVKEY',
                                                   'open_date',
                                                   'close_date',
                                                   'effective_date',
                                                   'trade_direction',
                                                   'adj_price',
                                                   'trade_size',
                                                   'market_value',
                                                   'weight',
                                                   'contribution',
                                                   'daily_return'
        })

    def open_trade(self, trade_info):
        self.port_holdings.append(trade_info)

    def close_trade(self, trade_info):
        self.port_holdings.append(trade_info)


class TradingStrategy:

    def __init__(self, trade_date):
        self.trade_date = trade_date


class BackTest:

    def __init__(self, X_data, y_data, beg_date, end_date, model_type, score_method):
        self.beg_date = pd.Timestamp(beg_date)
        self.end_date = pd.Timestamp(end_date)
        self.model_type = model_type
        self.score_method = score_method

        self.X_data = X_data
        self.y_data = y_data
        self.X_info_columns = ['GVKEY_asset1', 'GVKEY_asset2', 'prediction_date', 'evaluation_date']
        self.y_info_columns = ['prediction_date', 'evaluation_date', 'GVKEY_asset1', 'GVKEY_asset2',
                               'spread_return_60d_std', 'adjusted_price_asset1', 'adjusted_price_asset2', 'spread_t0']

        # Initialize ModelPipeline
        self.model = mdl.ModelPipeline(model_type, score_method)
        # Initialize a portfolio
        self.portfolio = Portfolio()

        # Define the dates when we need to retune hyperparameter or retrain model
        self.retune_hyperparam_dates, self.retrain_model_dates = find_dates(self.y_data.copy())

    def get_historical_data_for_given_date(self, trade_date: pd.Timestamp()):
        # Select historical dates data
        data_X = self.X_data.copy()
        data_y = self.y_data.copy()
        data_X = data_X[data_X['evaluation_date'] < trade_date]
        data_y = data_y[data_y['evaluation_date'] < trade_date]
        return data_X, data_y

    def get_the_most_updated_model(self, trade_date: pd.Timestamp()):
        # Get historical data & Drop the info columns
        X_data, y_data = self.get_historical_data_for_given_date(trade_date)

        # Check if we need to retune hyperparameters
        if trade_date in self.retune_hyperparam_dates.values:
            print(f"hyperparam retuning for trade date {trade_date}")
            self.model.hyperparameter_tunning(X_data, y_data)

        # Check if we need to retrain model today
        if trade_date in self.retrain_model_dates.values:
            print(f"retrain model for trade date {trade_date}")
            self.model.model_training(X_data, y_data)

    def make_prediction_for_trade_date(self, trade_date: pd.Timestamp()):
        print(f"making prediction for trade date {trade_date}")
        X_data = self.X_data.copy()
        X_data = X_data.loc[X_data['prediction_date'] == trade_date]

        # Make prediction
        y_pred, prob = self.model.get_prediction(X_data)
        return y_pred, prob

    def predict_for_all_dates(self):

        # Get the range of all trade dates
        range_mask = (self.y_data['prediction_date'] > self.beg_date) & (self.y_data['prediction_date'] < self.end_date)
        trade_date_range = self.y_data.copy().loc[range_mask, 'prediction_date'].unique()

        # Make prediction for each trade date within the range:
        for trade_date in trade_date_range:
            y_pred, prob = self.make_prediction_for_trade_date(trade_date)




if __name__ == '__main__':
    X_data, y_data = mdl.read_features_label_data()
    back_test = BackTest(X_data, y_data, beg_date='2012-01-03', end_date='2020-12-31', model_type='random_forest',
                         score_method='f1_macro')
    back_test.get_the_most_updated_model(trade_date='2012-01-03')
    y_predicted = back_test.make_prediction_for_trade_date(trade_date='2012-01-03')
    print(y_predicted)
