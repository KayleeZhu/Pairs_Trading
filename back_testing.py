import pandas as pd
import model_training as mdl
import datetime
import numpy as np
from pathlib import Path

import model_training as mdl
import trading_strategy as ts
import CONFIG


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

    def __init__(self, X_data, y_data, beg_date, end_date, model_type, score_method, param_dist_num, random_state):
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
        self.model = mdl.ModelPipeline(model_type, score_method, param_dist_num, random_state)

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

    def back_testing_given_period(self, prob_predicted_trade=0.5):

        # Get the range of all trade dates
        range_mask = (self.y_data['prediction_date'] >= self.beg_date) & (
                self.y_data['prediction_date'] <= self.end_date)
        trade_date_range = self.y_data.copy().loc[range_mask, 'prediction_date'].unique()

        # Make prediction for each trade date within the range & Update portfolio holdings
        for trade_date in trade_date_range:
            self.get_the_most_updated_model(trade_date)
            selected_trades = self.select_trades(trade_date, prob_predicted_trade)
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

    def calculate_monthly_returns(self):
        port = self.port_daily_returns.copy()
        port['year'] = port['effective_date'].dt.year
        port['month'] = port['effective_date'].dt.month
        port_monthly_returns = port.groupby(['year', 'month'])['daily_return'].apply(
            lambda x: (1 + x).prod() - 1).reset_index().rename(columns={'daily_return': 'monthly_return'})

        return port_monthly_returns

    def calculate_annual_returns(self):
        port = self.port_daily_returns.copy()
        port['year'] = port['effective_date'].dt.year

        port_annual_returns = port.groupby('year')['daily_return'].apply(lambda x: (1 + x).prod() - 1).reset_index().rename(
            columns={'daily_return': 'annual_return'})
        annual_std = port.groupby('year')['daily_return'].apply(
            lambda x: x.std() * np.sqrt(x.count())).reset_index().rename(columns={'daily_return': 'annual_std'})
        # TODO: think about if we should use sqrt(252) to replace,
        #       the current logic will overstate SR if some days have no holdings

        port_annual_returns = port_annual_returns.merge(annual_std, on='year')
        port_annual_returns['sharpe'] = port_annual_returns['annual_return'] / port_annual_returns['annual_std']

        return port_annual_returns

    def annualized_risk_and_return(self):
        port_daily_returns = self.port_daily_returns.copy()
        total_return = (1 + port_daily_returns['daily_return']).prod() - 1
        num_years = port_daily_returns['effective_date'].max().year - port_daily_returns[
            'effective_date'].min().year + 1

        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        standard_deviation = port_daily_returns['daily_return'].std() * np.sqrt(252)
        sharpe = annualized_return / standard_deviation

        perform_dict = {'annualized_return': annualized_return,
                        'annualized_std': standard_deviation,
                        'annualized_sharpe': sharpe
                        }
        performance = pd.Series(perform_dict)

        print(f"annualized_return is {annualized_return}")
        print(f"std is {standard_deviation}")
        print(f"sharpe is {sharpe}")
        return performance


if __name__ == '__main__':

    # Record run time
    start_time = datetime.datetime.now()

    # Read features & labels data
    features, labels = mdl.read_features_label_data()

    # Start Backtesting
    back_test = BackTest(features, labels, beg_date='2013-01-01', end_date='2020-12-31', model_type=CONFIG.model,
                         score_method='f1_macro',  param_dist_num=CONFIG.param_dist,
                         random_state=CONFIG.random_state_num
                         )
    back_test.back_testing_given_period(prob_predicted_trade=CONFIG.prob_predicted_trade)
    port = back_test.portfolio.port_holdings  # This gives you the portfolio holdings
    daily_returns = back_test.calculate_daily_returns()  # This gives daily return series
    monthly_returns = back_test.calculate_monthly_returns()  # This gives monthly return series
    annual_returns = back_test.calculate_annual_returns()  # This gives annual return series
    total_performance = back_test.annualized_risk_and_return()  # This gives performance over the backtesting period

    # Save all the information to backtest_result folder
    backtest_folder = Path('backtest_results') / Path(CONFIG.model)

    # Save Cumulative return plot
    fig = daily_returns.plot(x='effective_date', y='cum_return').get_figure()
    fig.savefig(backtest_folder / Path(f'return_plot_{CONFIG.tag_for_current_run}.pdf'))

    # Save all performance info
    daily_returns.to_csv(backtest_folder / Path(f'daily_returns_{CONFIG.tag_for_current_run}.csv'))
    monthly_returns.to_csv(backtest_folder / Path(f'monthly_returns_{CONFIG.tag_for_current_run}.csv'))
    annual_returns.to_csv(backtest_folder / Path(f'annual_returns_{CONFIG.tag_for_current_run}.csv'))
    total_performance.to_csv(backtest_folder / Path(f'total_performance_{CONFIG.tag_for_current_run}.csv'))

    # Save Portfolio holdings for investigation
    port.to_csv(backtest_folder / Path(f'port_holdings_{CONFIG.tag_for_current_run}.csv'))

    # Record run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
