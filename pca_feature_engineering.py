import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime

import CONFIG


class PCAFeatures:

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
        features = features.dropna(subset=[last_col])

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

    fea = PCAFeatures(data, historical_days)

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

    start_time = datetime.datetime.now()
    print("start working on PCA")

    all_features = get_all_features_for_pca(data, historical_days, features_list)
    exp_ratio, pca_features = apply_pca(all_features, num_components)

    # Convert the pca features array into a DF:
    pca_columns = []
    for i in range(1, num_components+1):
        pca_columns.append('pca' + str(i))
    pca_features = pd.DataFrame(data=pca_features, index=all_features.index, columns=pca_columns)

    # Join the PCA features to the original data
    pca_columns = ['GVKEY', 'date'] + pca_columns
    pca_features = pca_features.join(data)[pca_columns]

    # Recording time and notify user how well the PCA is doing
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
    print(f"The explained variance ratio is: {exp_ratio}")
    print(f"The sum of explained ratio is: {exp_ratio.sum()}")

    return pca_features, exp_ratio


if __name__ == '__main__':

    data_path = Path(f'data/1_cleaned_data/{CONFIG.cleaned_pkl_file_name}')
    df = pd.read_pickle(data_path)

    # Parameters Control:
    feature_list = ['return', 'cum_return', 'volume', 'current_eps', 'dividend_yield']
    pca_results, explained_ratio = generate_pca_features_for_clustering(data=df, num_components=4, historical_days=20,
                                                                        features_list=feature_list)

    pca_results.to_pickle(f'data/2_pca_features/pca_features.pkl')
