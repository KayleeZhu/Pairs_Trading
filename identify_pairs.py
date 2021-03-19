import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import generate_features as fea
import asset_clustering as ac


def get_cluster_list(asset_cluster):
    # num_assets = asset_cluster['cluster_label'].value_counts()

    cluster_label_list = asset_cluster['cluster_label'].unique()
    cluster_list = []
    for label in cluster_label_list:
        cluster = asset_cluster[asset_cluster.cluster_label == label]
        # cluster['num_assets'] = num_assets[label]
        cluster_list.append(cluster)

    return cluster_list


def get_possible_pairs_for_each_cluster(asset_cluster):
    cluster_list = get_cluster_list(asset_cluster)

    # Loop through all the clusters, get all possible pairs in each cluster and save as a list
    possible_pairs_for_each_cluster_list = []
    for one_cluster in cluster_list:
        all_possible_pairs_in_one_cluster = pd.DataFrame(
            [key for key in combinations(one_cluster['GVKEY'].unique(), 2)], columns=('asset_1', 'asset_2'))
        possible_pairs_for_each_cluster_list.append(all_possible_pairs_in_one_cluster)

    return possible_pairs_for_each_cluster_list


def cointegration_test_for_one_pair(all_data, one_pair_of_asset, training_date, num_training_days):

    # Get the correct time range
    ending_date = training_date
    beginning_date = datetime.strftime(
        (dt.timedelta(days=-num_training_days) + datetime.strptime(ending_date, '%Y-%m-%d')), '%Y-%m-%d')

    # Select the data with the targeted date range, from beginning_date to ending_date
    data_with_targeted_date_range = all_data.loc[(all_data.date >= beginning_date) & (all_data.date <= ending_date)]

    # Get the price series for asset 1 & 2
    price_series_asset1 = data_with_targeted_date_range.loc[
        data_with_targeted_date_range.GVKEY == one_pair_of_asset[0], ['GVKEY', 'date', 'adjusted_price']]
    price_series_asset2 = data_with_targeted_date_range.loc[
        data_with_targeted_date_range.GVKEY == one_pair_of_asset[1], ['GVKEY', 'date', 'adjusted_price']]

    # Apply cointegration test on two price series. Null is no cointegrated relationship
    # If p_value is < 0.05 then the pair is conintegrated
    _, p_value, _, = statsmodels.tsa.stattools.coint(price_series_asset1['adjusted_price'],
                                                     price_series_asset2['adjusted_price'])

    return {'asset1_gvkey': one_pair_of_asset[0],
            'asset2_gvkey': one_pair_of_asset[1],
            'training_date': training_date,
            'p_value': p_value
            }


def loop_through_cluster_list(asset_cluster, all_data, training_date, num_training_days):
    possible_pairs_list = get_possible_pairs_for_each_cluster(asset_cluster)
    coint_results_list = []
    pairs_counter = 0

    for pairs_to_check in possible_pairs_list:

        pairs_counter_one_cluster = 0
        for i in range(0, len(pairs_to_check)):
            each_pair = pairs_to_check.iloc[i]
            coint_dict = cointegration_test_for_one_pair(all_data, each_pair, training_date, num_training_days)
            coint_results_list.append(coint_dict)

            # See how many pairs we have for one day -- not final selection
            p_value = coint_dict['p_value']
            if p_value < 0.05:
                pairs_counter_one_cluster += 1
        print(
            f"cointegration test finished for {len(pairs_to_check)} possible pairs, "
            f"and {pairs_counter_one_cluster} pairs are found in the current cluster, ")
        pairs_counter += pairs_counter_one_cluster

    print(f"Total pairs for all clusters: {pairs_counter}")
    return coint_results_list


if __name__ == '__main__':

    data_path = Path('data/cleaned_data.pkl')
    dt = pd.read_pickle(data_path)

    # Get features
    feature_list = ['return', 'cum_return', 'volume', 'current_eps', 'dividend_yield']
    num_components = 4
    historical_days = 20
    pca_results, explained_ratio = fea.generate_pca_features_for_clustering(dt, num_components, historical_days,
                                                                            features_list=feature_list)

    # Asset clustering
    cluster = ac.Cluster('2020-12-31')
    pca_for_the_date = cluster.optics(pca_results)
    asset_cluster = ac.get_final_cluster(pca_for_the_date)

    # Identify pairs
