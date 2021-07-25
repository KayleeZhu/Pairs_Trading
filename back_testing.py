import pandas as pd
import model_training as mdl
import datetime
import numpy as np


import model_training as mdl
import trading_strategy as ts


def find_dates(beg_date, end_date, data_y):
    # Get the data between given beg_date & end_date
    range_mask = (data_y['prediction_date'] >= beg_date) & (data_y['prediction_date'] <= end_date)
    data = data_y.copy().loc[range_mask, :]

    # Find the first date in each year & first date in each month
    data['year'] = data['prediction_date'].dt.year
    data['month'] = data['prediction_date'].dt.month
    hyperparameter_retune_dates = data.groupby('year')['prediction_date'].min()
    model_retrain_dates = data.groupby(['year', 'month'])['prediction_date'].min()

    return hyperparameter_retune_dates, model_retrain_dates


def get_last_bus_date(data_y, today):
    # Get the last business of today based on dates record in data_y
    all_dates = data_y.copy()['prediction_date'].unique()
    today_index = np.where(all_dates == today)
    last_bus_date = all_dates[today_index[0][0] - 1]

    return last_bus_date


class BackTest:

    def __init__(self, X_data, y_data, beg_date, end_date, model_type, score_method):
        self.beg_date = pd.Timestamp(beg_date)
        self.end_date = pd.Timestamp(end_date)
        self.model_type = model_type
        self.score_method = score_method

        # Initialize features & labels data
        self.X_data = X_data
        self.y_data = y_data
        self.X_info_columns = ['GVKEY_asset1', 'GVKEY_asset2', 'prediction_date', 'evaluation_date']
        self.y_info_columns = ['prediction_date', 'evaluation_date', 'GVKEY_asset1', 'GVKEY_asset2',
                               'spread_return_60d_std',
                               'spread_t0', 'spread_t1', 'spread_t2', 'spread_t3', 'spread_t4', 'spread_t5',
                               'spread_return_1d', 'spread_return_2d', 'spread_return_3d', 'spread_return_4d',
                               'spread_return_5d']

        # Initialize ModelPipeline
        self.model = mdl.ModelPipeline(model_type, score_method)

        # Initialize a portfolio
        self.portfolio = ts.Portfolio(self.y_data)

        # Define the dates when we need to retune hyperparameter or retrain model
        self.retune_hyperparam_dates, self.retrain_model_dates = find_dates(self.beg_date, self.end_date,
                                                                                   self.y_data.copy())

    def get_historical_data_for_given_date(self, trade_date):
        # Select historical dates data
        data_X = self.X_data.copy()
        data_y = self.y_data.copy()
        data_X = data_X[data_X['evaluation_date'] < trade_date]
        data_y = data_y[data_y['evaluation_date'] < trade_date]
        return data_X, data_y

    def get_the_most_updated_model(self, trade_date):
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

    def make_prediction_for_trade_date(self, trade_date):
        print(f"making prediction for trade date {trade_date}")
        X_data = self.X_data.copy()
        X_data = X_data.loc[X_data['prediction_date'] == trade_date]
        y_data = self.y_data.copy()
        y_data = y_data.loc[y_data['prediction_date'] == trade_date]

        # Make prediction
        y_data = self.model.get_prediction(X_data, y_data)
        return y_data

    def select_trades(self, trade_date, required_prob):
        # Drop the trades with hold prediction --> only care about long / short
        prediction = self.make_prediction_for_trade_date(trade_date)
        selected_trades = prediction.loc[prediction['y_pred'] != 0, :].copy()

        # Drop the prediction where predicted probability of the label is less than given required_prob parameter
        prob_mask = selected_trades[['prob_short', 'prob_hold', 'prob_long']].max(axis=1) >= required_prob
        selected_trades = selected_trades.loc[prob_mask, :]

        # Drop the trades where spread is too small --> might results in very volatile condition
        # TODO: think a better way to address this issue
        spread_size_mask = selected_trades['spread_t0'].abs() >= 5
        selected_trades = selected_trades.loc[spread_size_mask, :]

        # Get only the useful info:
        selected_trades = selected_trades[
            ['prediction_date', 'GVKEY_asset1', 'GVKEY_asset2', 'spread_t0', 'spread_return_60d_std', 'y_pred']]
        selected_trades = selected_trades.rename(columns={'prediction_date': 'open_date', 'spread_t0': 'spread_price',
                                                          'y_pred': 'direction'})
        return selected_trades

    def back_testing_given_period(self):  # Change this method name to back test

        # Get the range of all trade dates
        range_mask = (self.y_data['prediction_date'] >= self.beg_date) & (
                self.y_data['prediction_date'] <= self.end_date)
        trade_date_range = self.y_data.copy().loc[range_mask, 'prediction_date'].unique()

        # Make prediction for each trade date within the range & Update portfolio holdings
        for trade_date in trade_date_range:
            self.get_the_most_updated_model(trade_date)
            selected_trades = self.select_trades(trade_date, 0.5)
            self.portfolio.update_port_holdings(trade_date, selected_trades)

    def calculate_daily_returns(self):
        port = self.portfolio.port_holdings.copy()
        port_daily_returns = (port.groupby('effective_date')[['contribution']]
                              .sum()
                              .reset_index()
                              .rename(columns={'contribution': 'daily_return'})
                              )
        port_daily_returns['cum_return'] = (port_daily_returns['daily_return'] + 1).cumprod() - 1
        self.port_daily_returns = port_daily_returns
        return port_daily_returns

    def risk_and_return(self):
        port_daily_returns = self.port_daily_returns.copy()
        total_return = (1 + port_daily_returns['daily_return']).prod() - 1
        num_years = port_daily_returns['effective_date'].max().year - port_daily_returns[
            'effective_date'].min().year + 1
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        standard_deviation = port_daily_returns['daily_return'].std() * np.sqrt(252)
        sharpe = annualized_return / standard_deviation

        print(f"annualized_return is {annualized_return}")
        print(f"std is {standard_deviation}")
        print(f"sharpe is {sharpe}")
        return sharpe


if __name__ == '__main__':
    features, labels = mdl.read_features_label_data()
    # Record run time
    start_time = datetime.datetime.now()

    # # Initialize BackTest object
    # model_list = ['logistic',  'decision_tree', 'random_forest']
    # for model in model_list:
    #     back_test = BackTest(features, labels, beg_date='2013-01-01', end_date='2020-12-31', model_type=model,
    #                          score_method='f1_macro')
    #     back_test.back_testing_given_period()
    #     port = back_test.portfolio.port_holdings  # --> This gives you the portfolio holdings
    #     port_returns = back_test.calculate_daily_returns()
    #     sharpe_ratio = back_test.risk_and_return()
    #
    #     fig = port_returns.plot(x='effective_date', y='cum_return').get_figure()
    #     fig.savefig('cum_return_plot' + model + '.pdf')
    #
    #     port_returns.to_csv('daily_returns' + model + '.csv')
    #     port.to_csv('port_holdings' + model + '.csv')

    # Define Model Type
    model = 'logistic'
    # model = 'decision_tree'
    # model = 'random_forest'

    back_test = BackTest(features, labels, beg_date='2013-01-01', end_date='2020-12-31', model_type=model,
                         score_method='f1_macro')
    back_test.back_testing_given_period()
    port = back_test.portfolio.port_holdings  # --> This gives you the portfolio holdings
    port_returns = back_test.calculate_daily_returns()
    sharpe_ratio = back_test.risk_and_return()

    fig = port_returns.plot(x='effective_date', y='cum_return').get_figure()
    fig.savefig('cum_return_plot' + model + '.pdf')

    port_returns.to_csv('daily_returns' + model + '.csv')
    port.to_csv('port_holdings' + model + '.csv')


    # Record run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
