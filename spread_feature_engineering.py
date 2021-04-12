import pandas as pd
from pathlib import Path
import numpy as np
import datetime

import generate_pca_features as pca_fea
import asset_clustering as ac
import identify_pairs as ip


def get_crsp_data(read_path):
    crsp_data = pd.read_pickle(read_path)

    # Calculate returns, cum returns and dividend yield
    crsp_data = pca_fea.calculate_daily_returns(crsp_data)
    crsp_data = pca_fea.calculate_cumulative_returns(crsp_data)
    crsp_data = pca_fea.calculate_dividend_yield(crsp_data)

    crsp_data.sort_values(by=['GVKEY', 'date'], inplace=True)

    return crsp_data


def get_pairs_data(read_path):
    # Read data
    pairs = pd.read_pickle(read_path)

    # Drop p_value column
    pairs = pairs.drop(columns=['p_value'])

    # Reset index
    pairs = pairs.reset_index(drop=True)

    # Convert date to datetime
    pairs['training_date'] = pd.to_datetime(pairs['training_date'], format='%Y-%m-%d')
    return pairs


class SpreadFeature:

    # Features to calculate
    # Volume, div, eps: avg & std for each stock
    # spread = adj_price diff between two stocks
    # spread return

    def __init__(self, all_data, pairs):
        self.all_data = all_data
        self.pairs = pairs
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

        # Get the previous 20 days' adjusted price
        feature_name = 'adjusted_price'
        features_columns = ['GVKEY', 'date', feature_name]
        asset_data = self.all_data.copy()[features_columns]

        for i in range(shift_range+1):
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

        for i in range(shift_range+1):
            asset1_price_col = feature_name + '_t_' + str(i) + '_asset1'
            asset2_price_col = feature_name + '_t_' + str(i) + '_asset2'
            spread_col = 'spread_t_' + str(i)
            data_with_pairs[spread_col] = data_with_pairs[asset1_price_col] - data_with_pairs[asset2_price_col]
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

    def get_spread_features_for_pairs(self):
        """
        Calculate 5, 10, 15, 20 ... 60d avg spread return & its std
        :return: {DataFrame} -- Paris' 5, 10, 15, 20 ... 60d avg spread return & its std feature
        """
        shift_range = 60
        data_with_pairs = self.calculate_spread_return(shift_range)

        # Calculate spread return for 5, 10, 15, 20 days
        selected_column = ['date', 'GVKEY_asset1', 'GVKEY_asset2']
        num_days_list = np.arange(5, shift_range+5, 5)
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

        return data_with_pairs[selected_column]

    def get_all_features(self):
        feature_one_asset = self.get_features_for_one_asset(feature_target_list=['return', 'current_eps',
                                                                                 'volume', 'dividend_yield'])
        spread_returns = self.get_spread_features_for_pairs()

        # Add two helpful columns for CV
        feature_one_asset['prediction_date'] = feature_one_asset['date']
        feature_one_asset['evaluation_date'] = feature_one_asset.groupby(['GVKEY'])['date'].shift(-5)

        # Attach pairs info
        data_with_pairs = feature_one_asset.merge(self.pairs.copy(), how='left', left_on=["date", "GVKEY"],
                                                  right_on=["training_date", "asset1_gvkey"])
        # Attach second asset info
        data_with_pairs = data_with_pairs.merge(feature_one_asset.copy(), how='left', left_on=["date", "asset2_gvkey"],
                                                right_on=["date", "GVKEY"], suffixes=('_asset1', '_asset2'))
        # Keep the rows where there is pair
        data_with_pairs = data_with_pairs.copy().dropna(subset=['GVKEY_asset2'])

        # Drop columns
        data_with_pairs = data_with_pairs.drop(columns=['training_date', 'asset1_gvkey', 'asset2_gvkey'])

        # Join spread return features
        all_features = spread_returns.merge(data_with_pairs.copy(), how='left',
                                            left_on=["date", "GVKEY_asset1", "GVKEY_asset2"],
                                            right_on=["date", "GVKEY_asset1", "GVKEY_asset2"])

        # Drop the rows where 5d spread return is missing -- date >= 2020-12-24
        last_date = datetime.datetime.strptime('2020-12-24', '%Y-%m-%d')
        all_features = all_features[all_features['date'] < last_date]

        # Drop & Rename columns
        all_features = all_features.rename(columns={'prediction_date_asset1': 'prediction_date',
                                                    'evaluation_date_asset1': 'evaluation_date'})
        all_features = all_features.drop(columns=['date', 'prediction_date_asset2', 'evaluation_date_asset2'])

        # Sort by date
        all_features.sort_values(by=['prediction_date', 'GVKEY_asset1', 'GVKEY_asset2'], inplace=True)

        return all_features

    def generate_label_y(self, upper_threshold_factor, lower_threshold_factor):

        # Get spread_return_std_data
        selected_column = ['date', 'GVKEY_asset1', 'GVKEY_asset2', 'spread_return_60d_std']
        spread_return_std_data = self.spread_return_feature.copy()[selected_column]

        # Get future 5 day's adjusted price
        features_columns = ['GVKEY', 'date', 'adjusted_price']
        asset_data = self.all_data.copy()[features_columns]
        asset_data['adj_price_t1'] = asset_data.groupby(['GVKEY'])['adjusted_price'].shift(-1)
        asset_data['adj_price_t2'] = asset_data.groupby(['GVKEY'])['adjusted_price'].shift(-2)
        asset_data['adj_price_t3'] = asset_data.groupby(['GVKEY'])['adjusted_price'].shift(-3)
        asset_data['adj_price_t4'] = asset_data.groupby(['GVKEY'])['adjusted_price'].shift(-4)
        asset_data['adj_price_t5'] = asset_data.groupby(['GVKEY'])['adjusted_price'].shift(-5)

        # Add two helpful columns for CV
        asset_data['prediction_date'] = asset_data['date']
        asset_data['evaluation_date'] = asset_data.groupby(['GVKEY'])['date'].shift(-5)

        # Attach pairs info
        data_with_pairs = asset_data.merge(self.pairs.copy(), how='left', left_on=["date", "GVKEY"],
                                           right_on=["training_date", "asset1_gvkey"])
        # Attach second asset info
        data_with_pairs = data_with_pairs.merge(asset_data.copy(), how='left', left_on=["date", "asset2_gvkey"],
                                                right_on=["date", "GVKEY"], suffixes=('_asset1', '_asset2'))

        # Calculate spread & spread return from t0 - t5
        data_with_pairs = data_with_pairs.eval('spread_t0 = adjusted_price_asset1 - adjusted_price_asset2')
        for i in range(1, 6):
            price_1_col = 'adj_price_t' + str(i) + '_asset1'
            price_2_col = 'adj_price_t' + str(i) + '_asset2'
            spread_today = 'spread_t' + str(i)
            spread_yesterday = 'spread_t' + str(i-1)
            spread_daily_return_col = 'spread_daily_return_d' + str(i)
            # Calculate spread
            data_with_pairs[spread_today] = data_with_pairs[price_1_col] - data_with_pairs[price_2_col]
            # Calculate spread daily return
            data_with_pairs[spread_daily_return_col] = (data_with_pairs[spread_today] - data_with_pairs[
                spread_yesterday]) / data_with_pairs[spread_yesterday].abs()

        data_with_pairs = data_with_pairs.eval('spread_cum_return_3d = (spread_t3 - spread_t0) / abs(spread_t0)')
        data_with_pairs = data_with_pairs.eval('spread_cum_return_5d = (spread_t5 - spread_t0) / abs(spread_t0)')

        # Attach the spread return std info
        data_with_pairs = spread_return_std_data.merge(data_with_pairs.copy(), how='left',
                                                       left_on=["date", "GVKEY_asset1", "GVKEY_asset2"],
                                                       right_on=["date", "GVKEY_asset1", "GVKEY_asset2"])

        # Generate labels based on threshold
        # Intuition: Assume today the spread is stable (that's why we didnt take action),
        # in 5 days if spread return is above threshold then long today
        # if spread return go down too much then we should short today
        long_mask = (data_with_pairs['spread_cum_return_5d'] > upper_threshold_factor * data_with_pairs[
            'spread_return_60d_std']) | (
                                data_with_pairs['spread_cum_return_3d'] > upper_threshold_factor * data_with_pairs[
                            'spread_return_60d_std'])

        short_mask = (data_with_pairs['spread_cum_return_5d'] < -lower_threshold_factor * data_with_pairs[
            'spread_return_60d_std']) | (
                                 data_with_pairs['spread_cum_return_3d'] < -lower_threshold_factor * data_with_pairs[
                             'spread_return_60d_std'])
        data_with_pairs['y'] = 0
        data_with_pairs.loc[long_mask, 'y'] = 1
        data_with_pairs.loc[short_mask, 'y'] = -1

        # Drop & Rename columns 
        # data_with_pairs = data_with_pairs.drop(columns=['training_date', 'asset1_gvkey', 'asset2_gvkey'])
        data_with_pairs = data_with_pairs.rename(columns={'prediction_date_asset1': 'prediction_date',
                                                          'evaluation_date_asset1': 'evaluation_date'})
        selected_column = ['prediction_date', 'evaluation_date', 'GVKEY_asset1', 'GVKEY_asset2',
                           'spread_t0', 'spread_t1', 'spread_t2', 'spread_t3', 'spread_t4', 'spread_t5',
                           'spread_daily_return_d1', 'spread_daily_return_d2', 'spread_daily_return_d3',
                           'spread_daily_return_d4', 'spread_daily_return_d5', 'y']
        data_with_pairs = data_with_pairs[selected_column]

        # Drop the rows where 5d spread return is missing -- date >= 2020-12-24
        last_date = datetime.datetime.strptime('2020-12-24', '%Y-%m-%d')
        data_with_pairs = data_with_pairs[data_with_pairs['prediction_date'] < last_date]

        # Sort by date
        data_with_pairs.sort_values(by=['prediction_date', 'GVKEY_asset1', 'GVKEY_asset2'], inplace=True)

        # Set self.y as labels & Return it
        self.y = data_with_pairs
        return data_with_pairs


if __name__ == '__main__':
    # Get all_data & pairs data
    data_path = Path('data/cleaned_data.pkl')
    data = get_crsp_data(data_path)

    pairs_path = Path('data/pairs_for_all_days.pkl')
    pairs_data = get_pairs_data(pairs_path)

    spread_features = SpreadFeature(all_data=data, pairs=pairs_data)

    # features = spread_features.get_features_for_one_asset(feature_target_list=['return', 'current_eps',
    #                                                                            'volume', 'dividend_yield'])

    # data_with_pairs_info = spread_features.get_spread_features_for_pairs()

    # pairs_features = spread_features.get_all_features()
    # pairs_features.to_csv('pairs_test.csv')
    # print(pairs_features)

    pairs_features = spread_features.generate_label_y(upper_threshold_factor=0.8, lower_threshold_factor=0.8)

    X = spread_features.X
    y = spread_features.y
    print(X)
    print(y)
    print(y.groupby('y').count())

    y_csv_path = Path('data/pairs_label.csv')
    X_csv_path = Path('data/pairs_features.csv')
    y_pkl_path = Path('data/pairs_label.pkl')
    X_pkl_path = Path('data/pairs_features.pkl')

    y.to_csv(y_csv_path)
    X.to_csv(X_csv_path)
    y.to_pickle(y_pkl_path)
    X.to_pickle(X_pkl_path)
