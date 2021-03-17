import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calculate_daily_returns(data):
    """
    Calculate daily return of each stock
    """
    data.sort_values(by=['GVKEY', 'date'], inplace=True)

    # Calculate adjusted price which encounter stock split & dividend
    data.eval('adjusted_price = price_close / adjustment_factor * daily_total_return_factor', inplace=True)

    # Calculate returns using adjusted price
    data['return'] = data.groupby(['GVKEY'])['adjusted_price'].pct_change(fill_method='ffill')

    # Drop the rows where return is NA
    data.dropna(subset=['return'], inplace=True)
    return data


def calculate_cumulative_returns(data):
    """
    Calculate cumulative return of each stock
    """

    dt['return_factor'] = dt['return'] + 1
    dt['cum_return'] = dt.groupby(['GVKEY'])['return_factor'].cumprod() - 1

    # Drop the rows where return is NA
    data.dropna(subset=['cum_return'], inplace=True)
    return data


def calculate_rolling_returns(data):
    # Calculate rolling returns with 5 day windows
    dt['roll_return'] = dt.groupby(['GVKEY'])['return_factor'].rolling(5).apply(lambda x: x.prod()) - 1

    # Drop the rows where return is NA
    data.dropna(subset=['roll_return'], inplace=True)
    return data


def calculate_correlation(data):
    # Calculate the correlation matrix of the investment universe

    return data


def calculate_dividend_yield(data):
    """
    Calculate dividend yield of each stock
    :param data: The DataFrame we need to work on which contains annual dividend and close price
    :return: DataFrame with dividend_yield column
    """
    data.eval('dividend_yield = annual_dividend / price_close', inplace=True)
    return data


class Features:

    def __init__(self, data, historical_days):
        self.data = data
        self.historical_days = historical_days

    def create_features_columns(self, feature_name):
        # Create historical features columns
        features_columns = [feature_name]

        for i in range(1, self.historical_days+1):
            column_name = feature_name + '_t_' + str(i)
            features_columns.append(column_name)
            self.data[column_name] = self.data.groupby(['GVKEY'])[feature_name].shift(i)

        return self.data[features_columns]

    def drop_rows_with_na_features(self, feature_name):
        # Drop the rows with NA returns due to the shift
        features = self.create_features_columns(feature_name)
        last_col = feature_name + '_t_' + str(self.historical_days)
        features.dropna(subset=[last_col], inplace=True)

        return features

    def get_features(self, feature_name, scale=True):
        # Get the cleaned features with no NA
        features = self.drop_rows_with_na_features(feature_name)

        if scale:
            # Scale the features
            scaler = StandardScaler()
            scaler.fit(features)
            scaled_features = scaler.transform(features)

            # Convert the scaled features array into DataFrame
            features = pd.DataFrame(data=scaled_features, index=features.index, columns=features.columns)

        return features


def get_all_features_for_pca(data, historical_days, features_list):

    fea = Features(data, historical_days)

    # Get all the features that requested in the features_list
    all_features_list = []
    for i in range(0, len(features_list)):
        # TODO: convert the features list into a dictionary to indicate if we need to scale for this features,
        #  assume default=True for now
        current_features = fea.get_features(feature_name=features_list[i], scale=True)
        all_features_list.append(current_features)

    # Combine all features into one DF:
    return pd.concat(all_features_list, axis=1)


def apply_pca(all_features, num_components):
    pca = PCA(num_components)
    Xr = pca.fit(all_features)
    exp_ratio = Xr.explained_variance_ratio_
    pca_features = Xr.transform(all_features)

    return exp_ratio, pca_features


def generate_pca_features_for_clustering(data, num_components, historical_days, features_list):

    start_time = datetime.now()
    print("start working on PCA")

    all_features = get_all_features_for_pca(data, historical_days, features_list)
    exp_ratio, pca_features = apply_pca(all_features, num_components)

    # Convert the pca features array into a DF:
    pca_columns = []
    for i in range(1, num_components+1):
        pca_columns.append('pca' + str(i))
    pca_features = pd.DataFrame(data=pca_features, index=all_features.index, columns=pca_columns)

    # Join the PCA features to the original data
    pca_features = pca_features.join(data)[pca_columns]

    # Recording time and notify user how well the PCA is doing
    end_time = datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
    print(f"The explained variance ratio is: {exp_ratio}")

    return pca_features, exp_ratio


if __name__ == '__main__':

    # Instruction:
    # Please run data_cleaning.py first to the cleaned data
    # Drag the clean_data.pkl file under "data" folder

    data_path = Path('data/cleaned_data.pkl')
    dt = pd.read_pickle(data_path)
    dt = calculate_daily_returns(dt)
    dt = calculate_cumulative_returns(dt)
    dt = calculate_rolling_returns(dt)
    dt = calculate_dividend_yield(dt)

    # Parameters Control:
    features = ['return', 'volume', 'current_eps']
    pca_results, explained_ratio = generate_pca_features_for_clustering(data=dt, num_components=10, historical_days=20,
                                                                        features_list=features)
