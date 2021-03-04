import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calculate_returns(data):

    data.sort_values(by=['GVKEY', 'date'], inplace=True)

    # Calculate adjusted price which encounter stock split & dividend
    data.eval('adjusted_price = price_close / adjustment_factor * daily_total_return_factor', inplace=True)

    # Calculate returns using adjusted price
    data['return'] = data.groupby(['GVKEY'])['adjusted_price'].pct_change(fill_method='ffill')

    # Drop the rows where return is NA
    data.dropna(subset=['return'], inplace=True)
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


def apply_pca(pca_features, num_components):
    pca = PCA(num_components)
    Xr = pca.fit(pca_features)
    exp_ratio = Xr.explained_variance_ratio_

    results = Xr.transform(pca_features)  # TODO: rename this!
    # TODO: convert this into DataFrame

    return exp_ratio, results


def generate_pca_features_for_clustering(data, num_components, historical_days, features_list):

    start_time = datetime.now()
    print("start working on PCA")

    pca_features = get_all_features_for_pca(data, historical_days, features_list)
    exp_ratio, results = apply_pca(pca_features, num_components)

    # Recording time and notify user how well the PCA is doing
    end_time = datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
    print(f"The explained variance ratio is: {exp_ratio}")

    return exp_ratio, results


if __name__ == '__main__':

    # Instruction:
    # Please run data_cleaning.py first to the cleaned data
    # Drag the clean_data.pkl file under "data" folder

    data_path = Path('data/cleaned_data.pkl')
    dt = pd.read_pickle(data_path)
    dt = calculate_returns(dt)

    explained_ratio, pca_results = generate_pca_features_for_clustering(data=dt, num_components=10, historical_days=20,
                                                                features_list=['return', 'volume', 'current_eps'])
