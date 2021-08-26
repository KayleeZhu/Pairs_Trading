import pandas as pd
import datetime
import numpy as np
from pathlib import Path

from model_training import ModelPipeline
import trading_strategy as ts
from performance_analysis import Performance
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

    def __init__(self, X_data, y_data, vix_data, beg_date, end_date, model_type, score_method, param_dist_num, random_state):
        self.beg_date = pd.Timestamp(beg_date)
        self.end_date = pd.Timestamp(end_date)
        self.model_type = model_type
        self.score_method = score_method
        self.vix_data = vix_data

        # Initialize features & labels data
        self.X_data = X_data
        self.y_data = y_data
        self.X_info_columns = CONFIG.X_info_columns
        self.y_info_columns = CONFIG.y_info_columns

        # Initialize ModelPipeline
        self.model = ModelPipeline(model_type, score_method, param_dist_num, random_state)

        # Initialize a portfolio
        self.portfolio = ts.Portfolio(self.y_data)

        # Define the dates when we need to retune hyperparameter or retrain model
        self.retune_hyperparam_dates, self.retrain_model_dates = find_dates(self.beg_date, self.end_date,
                                                                            self.y_data.copy())

    def get_historical_data_for_given_date(self, trade_date, num_hist_years=5):
        # Select recent years historical dates data, num_hist_years given by parameter
        data_X = self.X_data.copy()
        data_y = self.y_data.copy()
        # beg_date = str(int(trade_date[0:4])-num_hist_years) + '-01-01'
        # beg_date = trade_date + pd.offsets.DateOffset(years=-num_hist_years)
        beg_date = trade_date + pd.Timedelta(days=-365*num_hist_years)
        data_X = data_X[(data_X['evaluation_date'] < trade_date) & (data_X['evaluation_date'] > beg_date)]
        data_y = data_y[(data_y['evaluation_date'] < trade_date) & (data_X['evaluation_date'] > beg_date)]
        return data_X, data_y

    def get_the_most_updated_model(self, trade_date):
        # Get historical data & Drop the info columns
        X_data, y_data = self.get_historical_data_for_given_date(trade_date, num_hist_years=5)

        # Check if we need to retune hyperparameters
        if trade_date in self.retune_hyperparam_dates.values:
            print(f"hyperparam retuning for trade date {trade_date}")
            self.model.hyperparameter_tunning(X_data, y_data)

        # Check if we need to retrain model today
        if trade_date in self.retrain_model_dates.values:
            print(f"retrain model for trade date {trade_date}")
            self.model.model_training(X_data, y_data, self.vix_data, higher_weight_factor=1.2)

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

    def get_performance_results(self):
        perf = Performance(self.portfolio.port_holdings.copy())

        daily_returns = perf.port_daily_returns  # This gives daily return series
        monthly_returns = perf.calculate_monthly_returns()  # This gives monthly return series
        annual_returns = perf.calculate_annual_returns()  # This gives annual return series
        performance_summary = perf.annualized_risk_and_return()  # This gives performance over the backtesting period

        # Save all performance info
        daily_returns.to_csv(CONFIG.daily_returns_path)
        monthly_returns.to_csv(CONFIG.monthly_returns_path)
        annual_returns.to_csv(CONFIG.annual_returns_path)
        performance_summary.to_csv(CONFIG.performance_summary_path)

        # Save Cumulative return plot
        fig = daily_returns.plot(x='effective_date', y='cum_return').get_figure()
        fig.savefig(CONFIG.return_plot_path)

        # Save portfolio holdings
        self.portfolio.port_holdings.to_csv(CONFIG.portfolio_holdings_path)


if __name__ == '__main__':
    # Record run time
    start_time = datetime.datetime.now()

    # Get features & labels data & VIX
    y = pd.read_pickle(CONFIG.pairs_label_data_path)  # labels
    X = pd.read_pickle(CONFIG.pairs_features_data_path)  # features
    vix = pd.read_csv(CONFIG.vix_path)

    # Start Backtesting
    back_test = BackTest(X_data=X,
                         y_data=y,
                         vix_data=vix,
                         beg_date=CONFIG.beg_date,
                         end_date=CONFIG.end_date,
                         model_type=CONFIG.model_type,
                         score_method=CONFIG.score_method,
                         param_dist_num=CONFIG.param_dist,
                         random_state=CONFIG.random_state_num
                         )
    back_test.back_testing_given_period(prob_predicted_trade=CONFIG.prob_predicted_trade)
    back_test.get_performance_results()

    # Record run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
