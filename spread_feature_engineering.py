import pandas as pd
from pathlib import Path
import numpy as np
import datetime

import CONFIG


def get_pairs_data(read_path):
    # Read data
    pairs = pd.read_pickle(read_path)
    # Drop p_value column
    pairs = pairs.drop(columns=['p_value'])
    # Reset index
    pairs = pairs.reset_index(drop=True)
    # Convert date to datetime
    pairs['training_date'] = pd.to_datetime(pairs['training_date'], format='%Y-%m-%d')

    pairs = pairs.sort_values(by='training_date')
    return pairs


class SpreadFeature:
    # Features to calculate
    # Volume, div, eps: avg & std for each stock
    # spread = adj_price diff between two stocks
    # spread return

    def __init__(self, all_data, pairs, max_holding_period_days, target_return):
        self.all_data = all_data
        self.pairs = pairs
        self.max_holding_period_days = max_holding_period_days
        self.target_return = target_return
        self.spread_return_feature = self.get_spread_features_for_pairs()
        self.X = self.get_all_features()
        self.y = pd.DataFrame()

    @staticmethod
    def rolling_calculations(all_features_df, feature_target, num):
        """
        Apply rolling calculation on feature_target and add the calculated features to all_features_df
        :param all_features_df:
        :param feature_target:
        :param num: Rolling window size
        :return: DataFrame with added features columns
        """
        avg_col_name = feature_target + '_' + str(num) + 'd_avg'
        std_col_name = feature_target + '_' + str(num) + 'd_std'
        all_features_df[avg_col_name] = all_features_df.groupby('GVKEY')[feature_target].apply(
            lambda x: x.rolling(num).mean())
        all_features_df[std_col_name] = all_features_df.groupby('GVKEY')[feature_target].apply(
            lambda x: x.rolling(num).std())

        return all_features_df

    def get_features_for_one_asset(self, feature_target_list):

        all_data = self.all_data.copy()
        all_data.sort_values(by=['GVKEY', 'date'], inplace=True)

        # Select all the target features
        col_list = ['GVKEY', 'date'] + feature_target_list
        all_features = all_data[col_list].copy()

        num_days_list = [5, 10, 15, 20]
        for feature in feature_target_list:
            # Do not calculate avg & std for current eps as it doesn't make sense
            if feature != 'current_eps':
                for num in num_days_list:
                    all_features = self.rolling_calculations(all_features, feature, num)

        return all_features

    def calculate_spread_return(self, shift_range):
        """  This method attach pairs information to all data,
            calculate adj price spread between two stocks for previous 20 days,
        """

        feature_name = 'adjusted_price'
        features_columns = ['GVKEY', 'date', feature_name]
        asset_data = self.all_data.copy()[features_columns]

        # Get the previous 20 days' adjusted price
        for i in range(shift_range + 1):
            column_name = feature_name + '_t_' + str(i)
            asset_data[column_name] = asset_data.groupby(['GVKEY'])[feature_name].shift(i)

        # Attach pairs info
        data_with_pairs = asset_data.merge(self.pairs.copy(), how='left', left_on=["date", "GVKEY"],
                                           right_on=["training_date", "asset1_gvkey"])
        # Attach second asset info
        data_with_pairs = data_with_pairs.merge(asset_data.copy(), how='left', left_on=["date", "asset2_gvkey"],
                                                right_on=["date", "GVKEY"], suffixes=('_asset1', '_asset2'))

        # Calculate spread = adj_price diff
        selected_column = ['date', 'GVKEY_asset1', 'GVKEY_asset2']

        for i in range(shift_range + 1):
            asset1_price_col = feature_name + '_t_' + str(i) + '_asset1'
            asset2_price_col = feature_name + '_t_' + str(i) + '_asset2'
            spread_col = 'spread_t_' + str(i)
            # data_with_pairs[spread_col] = data_with_pairs[asset1_price_col] - data_with_pairs[asset2_price_col]
            data_with_pairs[spread_col] = data_with_pairs[asset1_price_col] * data_with_pairs['asset1_weight'] - \
                                          data_with_pairs[asset2_price_col] * data_with_pairs['asset1_weight']
            # selected_column.append(spread_col)

        # Calculate spread return
        for i in range(shift_range):
            spread1_col = 'spread_t_' + str(i + 1)  # Previous day
            spread2_col = 'spread_t_' + str(i)  # Today
            return_col = 'spread_return_t_' + str(i)  # Return of today
            data_with_pairs[return_col] = (data_with_pairs[spread2_col] - data_with_pairs[spread1_col]) / \
                                          data_with_pairs[spread1_col]
            selected_column.append(return_col)

        # Select the features columns only and keep the rows where there is pair
        data_with_pairs = data_with_pairs[selected_column]
        data_with_pairs = data_with_pairs.copy().dropna(subset=['GVKEY_asset2'])
        data_with_pairs.sort_values(by=['date', 'GVKEY_asset1', 'GVKEY_asset2'], inplace=True)

        return data_with_pairs

    def get_spread_features_for_pairs(self, shift_range=60):
        """
        Calculate 5, 10, 15, 20 ... 60d avg spread return & its std
        :return: {DataFrame} -- Paris' 5, 10, 15, 20 ... 60d avg spread return & its std feature
        """
        data_with_pairs = self.calculate_spread_return(shift_range)

        # Calculate spread return for 5, 10, 15, 20 days
        selected_column = ['date', 'GVKEY_asset1', 'GVKEY_asset2']
        num_days_list = np.arange(5, shift_range + 5, 5)
        for num in num_days_list:
            avg_col_name = 'spread_return_' + str(num) + 'd_avg'
            std_col_name = 'spread_return_' + str(num) + 'd_std'
            selected_column.append(avg_col_name)
            selected_column.append(std_col_name)

            # Get the list of columns that we need to calculate avg or std on
            spread_col_list = []
            for i in range(num):
                spread_col_list.append('spread_return_t_' + str(i))

            data_with_pairs[avg_col_name] = data_with_pairs[spread_col_list].mean(axis=1)
            data_with_pairs[std_col_name] = data_with_pairs[spread_col_list].std(axis=1)

        return data_with_pairs[selected_column].copy()

    def get_all_features(self):
        print('start getting all features X')
        feature_one_asset = self.get_features_for_one_asset(feature_target_list=['return', 'current_eps',
                                                                                 'volume', 'dividend_yield'])
        spread_returns = self.get_spread_features_for_pairs(shift_range=60)

        # Add two helpful columns for CV
        feature_one_asset['prediction_date'] = feature_one_asset['date']
        feature_one_asset['evaluation_date'] = feature_one_asset.groupby(['GVKEY'])['date'].shift(
            -self.max_holding_period_days)

        # Attach pairs info
        data_with_pairs = feature_one_asset.merge(self.pairs, how='inner', left_on=["date", "GVKEY"],
                                                  right_on=["training_date", "asset1_gvkey"])
        # Attach second asset info
        data_with_pairs = data_with_pairs.merge(feature_one_asset, how='inner', left_on=["date", "asset2_gvkey"],
                                                right_on=["date", "GVKEY"], suffixes=('_asset1', '_asset2'))

        # Drop columns
        data_with_pairs = data_with_pairs.drop(columns=['training_date', 'asset1_gvkey', 'asset2_gvkey'])

        # Join spread return features
        all_features = spread_returns.merge(data_with_pairs, how='left',
                                            left_on=["date", "GVKEY_asset1", "GVKEY_asset2"],
                                            right_on=["date", "GVKEY_asset1", "GVKEY_asset2"])

        # Drop the rows where it has NA or inf values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.dropna(how='any')

        # Drop & Rename columns
        all_features = all_features.rename(columns={'prediction_date_asset1': 'prediction_date',
                                                    'evaluation_date_asset1': 'evaluation_date'})
        all_features = all_features.drop(columns=['date', 'prediction_date_asset2', 'evaluation_date_asset2'])

        # Sort by date
        all_features.sort_values(by=['prediction_date', 'GVKEY_asset1', 'GVKEY_asset2'], inplace=True)
        return all_features

    def get_returns_in_future_holding_period(self, data_with_pairs):
        # Calculate spread & spread daily return & cumulative return for future max_holding_period
        data_with_pairs = data_with_pairs.eval(
            'spread_t0 = adjusted_price_asset1 * asset1_weight - adjusted_price_asset2 * asset2_weight')

        # Spread & Spread daily return
        for i in range(1, self.max_holding_period_days + 1):
            price_1_col = 'adj_price_t' + str(i) + '_asset1'
            price_2_col = 'adj_price_t' + str(i) + '_asset2'
            spread_today = 'spread_t' + str(i)
            spread_yesterday = 'spread_t' + str(i - 1)
            spread_daily_return_col = 'spread_daily_return_d' + str(i)
            # Calculate spread
            data_with_pairs[spread_today] = data_with_pairs[price_1_col] * data_with_pairs['asset1_weight'] - \
                                            data_with_pairs[price_2_col] * data_with_pairs['asset2_weight']
            # Calculate spread daily return
            data_with_pairs[spread_daily_return_col] = (data_with_pairs[spread_today] - data_with_pairs[
                spread_yesterday]) / data_with_pairs[spread_yesterday].abs()

        # Spread Cumulative return
        for i in range(1, self.max_holding_period_days + 1):
            spread_t_col = 'spread_t' + str(i)
            cum_return_t_col = 'spread_cum_return_' + str(i) + 'd'
            data_with_pairs[cum_return_t_col] = (data_with_pairs[spread_t_col] - data_with_pairs['spread_t0']) / abs(
                data_with_pairs['spread_t0'])

        return data_with_pairs

    def triple_barrier(self, data_with_pairs):
        """ generate labels using triple_barrier rules, return long & short mask
            If in the future holding period, any date reaches the return target, then long today
            If any date reaches the -return target, then short
        """

        # Initialize labels as 0
        data_with_pairs['y'] = 0

        # Generate labels using path dependent triple barrier rules
        for i in range(1, self.max_holding_period_days + 1):
            cum_return_t_col = 'spread_cum_return_' + str(i) + 'd'
            zero_mask = data_with_pairs['y'] == 0
            long_mask = data_with_pairs[cum_return_t_col] >= self.target_return  # At time t, if reach target then long
            short_mask = data_with_pairs[cum_return_t_col] <= -self.target_return
            data_with_pairs.loc[
                (long_mask & zero_mask), 'y'] = 1  # only overwrites the prediction where it hasn't reached target
            data_with_pairs.loc[(short_mask & zero_mask), 'y'] = -1

        return data_with_pairs.copy()

    def generate_label_y(self, upper_threshold_factor, lower_threshold_factor):
        print('start generate_label_y')

        # Get spread_return_std_data
        selected_column = ['date', 'GVKEY_asset1', 'GVKEY_asset2', 'spread_return_60d_std']
        spread_return_std_data = self.spread_return_feature[selected_column].copy()

        # Get future daily adjusted price in future holding period days
        features_columns = ['GVKEY', 'date', 'adjusted_price']
        asset_data = self.all_data[features_columns].copy()
        for i in range(0, self.max_holding_period_days + 1):
            price_col = 'adj_price_t' + str(i)
            asset_data[price_col] = asset_data.groupby(['GVKEY'])['adjusted_price'].shift(-i)
        print('future adj price done')

        # Add two helpful columns for CV
        asset_data['prediction_date'] = asset_data['date']
        asset_data['evaluation_date'] = asset_data.groupby(['GVKEY'])['date'].shift(-self.max_holding_period_days)

        # Attach pairs info
        data_with_pairs = asset_data.merge(self.pairs, how='inner', left_on=['date', 'GVKEY'],
                                           right_on=['training_date', 'asset1_gvkey'])
        # Attach second asset info
        data_with_pairs = data_with_pairs.merge(asset_data, how='inner', left_on=['date', 'asset2_gvkey'],
                                                right_on=['date', 'GVKEY'], suffixes=('_asset1', '_asset2'))

        # Calculate spread & spread return for future holding period days
        data_with_pairs = self.get_returns_in_future_holding_period(data_with_pairs)
        print('future returns calculation done')

        # Attach the spread return std info
        data_with_pairs = spread_return_std_data.merge(data_with_pairs, how='left',
                                                       left_on=['date', 'GVKEY_asset1', 'GVKEY_asset2'],
                                                       right_on=['date', 'GVKEY_asset1', 'GVKEY_asset2'])

        # Generate labels using path dependent triple barrier rules
        data_with_pairs = self.triple_barrier(data_with_pairs)
        print('triple barrier done')

        # Drop & Rename columns 
        # data_with_pairs = data_with_pairs.drop(columns=['training_date', 'asset1_gvkey', 'asset2_gvkey'])
        data_with_pairs = data_with_pairs.rename(columns={'prediction_date_asset1': 'prediction_date',
                                                          'evaluation_date_asset1': 'evaluation_date'})
        selected_column = CONFIG.y_info_columns + ['y']
        data_with_pairs = data_with_pairs[selected_column]

        # Drop the rows where it has NA or inf values
        data_with_pairs = data_with_pairs.replace([np.inf, -np.inf], np.nan)
        data_with_pairs = data_with_pairs.dropna(how='any')

        # Sort by date
        data_with_pairs = data_with_pairs.sort_values(by=['prediction_date', 'GVKEY_asset1', 'GVKEY_asset2'])

        # Set self.y as labels & Return it
        self.y = data_with_pairs.copy()
        return data_with_pairs

    def check_X_y_dimensionality(self):
        """
        Make sure X & y has same dimensionality. If not, then overwrites it
        """
        if len(self.X) > len(self.y):  # if X longer, then only gets y's index
            self.X = self.X.iloc[self.y.index].copy()
        elif len(self.X) < len(self.y):
            self.y = self.y.iloc[self.X.index].copy()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("start generating spread features and labels")

    # Get cleaned data & pairs data
    cleaned_data = pd.read_pickle(CONFIG.cleaned_data_path)
    pairs_data = get_pairs_data(CONFIG.pairs_data_path)

    # Get spread features
    spread_features = SpreadFeature(all_data=cleaned_data,
                                    pairs=pairs_data,
                                    max_holding_period_days=CONFIG.max_holding_period_days,
                                    target_return=CONFIG.target_return
                                    )
    pairs_features = spread_features.generate_label_y(upper_threshold_factor=0.8,
                                                      lower_threshold_factor=0.8
                                                      )
    spread_features.check_X_y_dimensionality()

    X = spread_features.X
    y = spread_features.y
    print(X)
    print(y)
    print(y.groupby('y').count())

    # Save data
    y.to_pickle(CONFIG.pairs_label_data_path)
    X.to_pickle(CONFIG.pairs_features_data_path)

    # Run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
