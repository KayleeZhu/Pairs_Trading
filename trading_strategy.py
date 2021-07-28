import pandas as pd
import model_training as mdl
import datetime
import numpy as np
from scipy.optimize import minimize

import back_testing as bt
import CONFIG


class TradeExit:
    def __init__(self):
        pass

    def take_profits(self):
        pass

    def stop_loss(self):
        pass


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
        last_bus_day = bt.get_last_bus_date(self.market_data, update_date)
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

        # Portfolio Constructions:

        # Update weight & contribution using num_active_trades
        current_port['weight'] = current_port['spread_price'].abs() / current_port[
            'spread_price'].abs().sum()  # Market cap weighted
        # current_port['weight'] = 1 / current_port['num_active_trades']  # Equally weighted

        # Set a cap on weight --> Doesn't make sense to use all the money to make only a few trades
        current_port['weight'] = np.minimum(current_port['weight'], 0.2)
        current_port['contribution'] = current_port['spread_return'] * current_port['weight']

        # Close out trades --> Fill up close date col and set is_active to 0
        # Stop Loss condition
        # stop_loss_mask = current_port['cum_spread_return'] < -0.8 * current_port['spread_return_60d_std']
        # current_port.loc[stop_loss_mask, ['close_date']] = trade_date
        # current_port.loc[stop_loss_mask, ['is_active']] = 0

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


if __name__ == '__main__':
    features, labels = mdl.read_features_label_data()
    # Record run time
    start_time = datetime.datetime.now()
