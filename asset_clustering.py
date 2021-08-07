import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import datetime

import CONFIG


class Cluster:

    def __init__(self, training_date, pca_features):
        self.training_date = training_date
        self.pca_features = pca_features

    def kmeans(self, n_clusters):
        # TODO：need to choose n, see which n works better
        # TODO: reformat the code to make it has the same output as optics

        # Get the training date PCA features data
        pca_for_the_date = self.pca_features.loc[self.pca_features['date'] == self.training_date].copy()
        cluster = KMeans(n_clusters).fit(pca_for_the_date)

        # Join the cluster label to the PCA data for the training date
        pca_for_the_date = pca_for_the_date.join(
            pd.DataFrame(data=cluster.labels_, index=pca_for_the_date.index, columns=['cluster_label']))

        # Drop the PCA values (col) and  drop the assets with cluster label = -1 (outliers)
        cluster_label_all_assets = pca_for_the_date[['GVKEY', 'date', 'cluster_label']]
        cluster_label_all_assets = cluster_label_all_assets[cluster_label_all_assets.cluster_label != -1]
        # print(cluster.labels_)

        return cluster_label_all_assets

    def optics(self):
        # Get the training date PCA features data
        pca_for_the_date = self.pca_features.loc[self.pca_features['date'] == self.training_date].copy()
        cluster = OPTICS().fit(pca_for_the_date[[col for col in pca_for_the_date.columns if 'pca' in col]])

        # Join the cluster label to the PCA data for the training date
        pca_for_the_date = pca_for_the_date.join(
            pd.DataFrame(data=cluster.labels_, index=pca_for_the_date.index, columns=['cluster_label']))

        # Drop the PCA values (col) and  drop the assets with cluster label = -1 (outliers)
        cluster_label_all_assets = pca_for_the_date[['GVKEY', 'date', 'cluster_label']]
        cluster_label_all_assets = cluster_label_all_assets[cluster_label_all_assets.cluster_label != -1]
        # print(cluster.labels_)

        # TODO: how to visualize the cluster?
        return cluster_label_all_assets


def get_monthly_cluster_dates(beg_date, end_date, data):
    # Get the data between given beg_date & end_date
    range_mask = (data['date'] >= beg_date) & (data['date'] <= end_date)
    data = data.copy().loc[range_mask, :]

    # Find the first date in each year & first date in each month
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    month_beg = data.groupby(['year', 'month'])['date'].min()
    return month_beg


def get_last_bus_date(data, today):
    # Get the last business of today based on dates record in data_y
    all_dates = data.copy()['date'].unique()
    today_index = np.where(all_dates == today)
    last_bus_date = all_dates[today_index[0][0] - 1]

    return last_bus_date


def generate_cluster_for_all_dates(beg_date, end_date, pca_features, retrain_freq='daily'):

    # Assume by default, we retrain cluster daily
    range_mask = (pca_features['date'] >= beg_date) & (pca_features['date'] <= end_date)
    data_in_range = pca_features.copy().loc[range_mask, :]
    cluster_date_list = data_in_range['date'].unique()
    if retrain_freq == 'monthly':
        cluster_date_list = get_monthly_cluster_dates(beg_date, end_date, pca_features)

    cluster_list = []
    for date in cluster_date_list:
        print(f'asset clustering for date {date}')
        asset_cluster = Cluster(date, pca_features)
        asset_cluster = asset_cluster.optics()
        cluster_list.append(asset_cluster)

    return pd.concat(cluster_list, axis=0)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("start working on asset clustering")

    data_path = Path(f'data/1_cleaned_data/{CONFIG.cleaned_pkl_file_name}')
    cleaned_data = pd.read_pickle(data_path)

    # Get PCA features for clustering from data folder
    pca_results = pd.read_pickle(f'data/2_pca_features/pca_features.pkl')

    # Clustering
    # asset_cluster = Cluster(training_date='2020-12-31', pca_features=pca_results)
    # asset_cluster = asset_cluster.optics()

    # Generated asset cluster for all dates
    training_freq = 'daily'
    clusters_for_all_dates = generate_cluster_for_all_dates(beg_date='2000-01-04', end_date='2020-12-31',
                                                            pca_features=pca_results, retrain_freq=training_freq)
    clusters_for_all_dates.to_pickle(f'data/3_asset_cluster/clusters_for_all_dates_{training_freq}.pkl')

    # Run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
