import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
import generate_features as fea


class Cluster:

    def __init__(self, training_date):
        self.training_date = training_date

    def knn(self, pca_features, n_neighbors):
        # TODO：need to choose n, see which n works better
        neigh = NearestNeighbors(n_neighbors)
        neigh.fit(pca_features)

    def optics(self, pca_features):
        # Get the training date PCA features data
        pca_for_the_date = pca_features.loc[pca_features['date'] == self.training_date]
        pca_for_the_date = pca_for_the_date[[col for col in pca_for_the_date.columns if 'pca' in col]]
        cluster = OPTICS().fit(pca_for_the_date)

        # Get the training date PCA features data
        pca_for_the_date = pca_results.loc[pca_results['date'] == self.training_date]
        # pca_for_the_date = pca_for_the_date[[col for col in pca_for_the_date.columns if 'pca' in col]]
        pca_for_the_date = pca_for_the_date.join(
            pd.DataFrame(data=cluster.labels_, index=pca_for_the_date.index, columns=['cluster_label']))

        print(cluster.labels_)

        # TODO: how to visualize the cluster?

        return pca_for_the_date


def get_final_cluster(pca_for_the_date):
    # Drop the assets with cluster label = -1 (outliers)
    pca_for_the_date = pca_for_the_date[['GVKEY', 'date', 'cluster_label']]
    pca_for_the_date = pca_for_the_date[pca_for_the_date.cluster_label != -1]

    return pca_for_the_date


if __name__ == '__main__':

    data_path = Path('data/cleaned_data.pkl')
    dt = pd.read_pickle(data_path)

    # Parameters Control:
    feature_list = ['return', 'cum_return', 'volume', 'current_eps', 'dividend_yield']
    pca_results, explained_ratio = fea.generate_pca_features_for_clustering(data=dt, num_components=4, historical_days=20,
                                                                            features_list=feature_list)

    cluster = Cluster('2020-12-31')
    pca_for_the_date = cluster.optics(pca_results)
    asset_cluster = get_final_cluster(pca_for_the_date)
