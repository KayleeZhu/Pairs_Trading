import pandas as pd
import model_training as mdl
import datetime
import numpy as np


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


class Portfolio:
    # write a portfolio DataFrame, whenever we enter a trade, add a row here
    def __init__(self, market_data):
        self.port_columns = ['trade_id',
                             'effective_date',
                             'GVKEY_asset1',
                             'GVKEY_asset2',
                             'open_date',
                             'close_date',
                             'holding_period',
                             'direction',
                             'spread_price',
                             'spread_return',
                             'cum_spread_return',
                             'spread_return_60d_std',
                             'is_active',
                             'num_active_trades',
                             'weight',
                             'contribution'
                             ]
        self.port_holdings = pd.DataFrame(columns=self.port_columns)
        self.market_data = market_data

    def update_price_and_return(self, current_port, update_date):
        # Find the market_data for current_port holdings
        market = self.market_data.copy()

        # Join market info to current_port
        current_port = current_port.merge(market, how='left', left_on=['trade_id', 'open_date'],
                                          right_on=[market.index, 'prediction_date'])

        # Loop through holding period 1-5 and fill the updated spread price and return info
        for i in range(1, 6):
            holding_period_mask = current_port['holding_period'] == i
            spread_col_name = 'spread_t' + str(i)
            spread_return_col = 'spread_daily_return_d' + str(i)
            current_port.loc[holding_period_mask, ['spread_price']] = current_port.loc[holding_period_mask,
                                                                                       spread_col_name]
            current_port.loc[holding_period_mask, ['spread_return']] = current_port.loc[holding_period_mask,
                                                                                        spread_return_col]

        current_port['cum_spread_return'] = (1 + current_port['cum_spread_return']) * (
                1 + current_port['spread_return']) - 1
        current_port = current_port.set_index('trade_id')
        return current_port[['spread_price', 'spread_return', 'cum_spread_return']]

    def update_market_data(self, update_date):
        # Make a copy of yesterday's port_holdings
        last_bus_day = get_last_bus_date(self.market_data, update_date)
        last_bus_day_mask = self.port_holdings['effective_date'] == last_bus_day
        active_holdings_mask = self.port_holdings['is_active'] == 1

        # Take yesterday's active portfolio holdings
        current_port = self.port_holdings.copy()[last_bus_day_mask & active_holdings_mask]

        # Only update when there is holdings
        if len(current_port) != 0:
            # Update some columns of port_last_bus_day
            current_port['effective_date'] = update_date
            current_port['holding_period'] = current_port['holding_period'] + 1

            # Use market_data to update today's price & return
            updated_info = self.update_price_and_return(current_port, update_date)
            current_port[['spread_price', 'spread_return', 'cum_spread_return']] = updated_info[
                ['spread_price', 'spread_return', 'cum_spread_return']]

            # Adjust the sign
            current_port['spread_return'] = current_port['spread_return'] * current_port['direction']
            current_port['cum_spread_return'] = current_port['cum_spread_return'] * current_port['direction']
            self.port_holdings = self.port_holdings.append(current_port)

    def open_trade(self, selected_trades):
        # Add columns for selected_trades
        selected_trades['trade_id'] = selected_trades.index
        selected_trades['effective_date'] = selected_trades['open_date']
        selected_trades['close_date'] = pd.NaT
        selected_trades['holding_period'] = 0
        selected_trades['spread_return'] = 0
        selected_trades['cum_spread_return'] = 0
        selected_trades['is_active'] = 1
        selected_trades['num_active_trades'] = len(selected_trades)
        selected_trades['weight'] = 1 / selected_trades['num_active_trades']
        selected_trades['contribution'] = selected_trades['spread_return'] * selected_trades['weight']

        self.port_holdings = self.port_holdings.append(selected_trades)

    def close_trade(self, trade_date):
        active_holdings_mask = self.port_holdings['is_active'] == 1
        today_mask = self.port_holdings['effective_date'] == trade_date
        current_port = self.port_holdings.copy()[today_mask & active_holdings_mask]

        # Update the num_active_trades
        current_port['num_active_trades'] = len(current_port)
        # Update weight & contribution using num_active_trades
        current_port['weight'] = current_port['spread_price'].abs() / current_port['spread_price'].abs().sum()  # Market cap weighted
        # current_port['weight'] = 1 / current_port['num_active_trades']  # Equally weighted

        # Set a cap on weight --> Doesn't make sense to use all the money to make only a few trades
        current_port['weight'] = np.minimum(current_port['weight'], 0.2)

        current_port['contribution'] = current_port['spread_return'] * current_port[
            'weight']

        # Close out trades --> Fill up close date col and set is_active to 0
        # Stop Loss condition
        stop_loss_mask = current_port['cum_spread_return'] < -0.8 * current_port['spread_return_60d_std']
        current_port.loc[stop_loss_mask, ['close_date']] = trade_date
        current_port.loc[stop_loss_mask, ['is_active']] = 0

        # Check at d3 if we get what the model predicts, if yes then exit trades
        close_mask_3d = current_port['holding_period'] == 3
        close_mask = current_port['cum_spread_return'] > 0.8 * current_port['spread_return_60d_std']
        current_port.loc[close_mask_3d & close_mask, ['close_date']] = trade_date
        current_port.loc[close_mask_3d & close_mask, ['is_active']] = 0

        # Exit trade at d5 no matter what
        close_mask_5d = current_port['holding_period'] == 5
        current_port.loc[close_mask_5d, ['close_date']] = trade_date
        current_port.loc[close_mask_5d, ['is_active']] = 0

        # Overwrite new data in port_holdings
        self.port_holdings.loc[today_mask & active_holdings_mask] = current_port

    def update_port_holdings(self, trade_date, selected_trades):
        # 1. update market data
        self.update_market_data(trade_date)
        # 2. open trades
        self.open_trade(selected_trades)
        # 3. close trades
        self.close_trade(trade_date)


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
        self.portfolio = Portfolio(self.y_data)

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

    # Initialize BackTest object
    model_list = ['logistic',  'decision_tree', 'random_forest']
    for model in model_list:
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
