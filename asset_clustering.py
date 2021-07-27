import pandas as pd
from pathlib import Path
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import pca_feature_engineering as fea

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


if __name__ == '__main__':

    data_path = Path(f'data/cleaned_data_{CONFIG.sectors_num}_sectors.pkl')
    dt = pd.read_pickle(data_path)

    # Parameters Control --> Get PCA features for clustering
    feature_list = ['return', 'cum_return', 'volume', 'current_eps', 'dividend_yield']
    pca, explained_ratio = fea.generate_pca_features_for_clustering(data=dt, num_components=4, historical_days=20,
                                                                    features_list=feature_list)

    # Clustering
    asset_cluster = Cluster(training_date='2020-12-31', pca_features=pca)
    asset_cluster = asset_cluster.optics()
