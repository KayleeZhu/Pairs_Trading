import pandas as pd
from pathlib import Path
import datetime
from itertools import combinations
from joblib import Parallel, delayed

import statsmodels
from statsmodels.tsa.stattools import coint

import generate_pca_features as fea
import asset_clustering as ac
import functools


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
    beginning_date = datetime.datetime.strftime(
        (datetime.timedelta(days=-num_training_days) + datetime.datetime.strptime(ending_date, '%Y-%m-%d')), '%Y-%m-%d')

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

    return {'training_date': training_date,
            'asset1_gvkey': one_pair_of_asset[0],
            'asset2_gvkey': one_pair_of_asset[1],
            'p_value': p_value
            }


def loop_through_cluster_list(all_data, asset_cluster, training_date, num_training_days, significant_level=0.05):
    # Loop through the cluster list and get all pairs for the given trading date
    # Return a dictionary storing info of all possible paris --> coint_results_list
    # Return a DataFrame storing all true pairs for the given training_date
    possible_pairs_list = get_possible_pairs_for_each_cluster(asset_cluster)
    coint_results_list = []
    true_pairs_list = []
    pairs_counter = 0

    for pairs_to_check in possible_pairs_list:

        pairs_counter_one_cluster = 0
        for i in range(0, len(pairs_to_check)):
            one_pair = pairs_to_check.iloc[i]
            coint_dict = cointegration_test_for_one_pair(all_data, one_pair, training_date, num_training_days)
            coint_results_list.append(coint_dict)

            # See how many pairs we have for one day -- not final selection
            p_value = coint_dict['p_value']
            if p_value < significant_level:
                true_pairs_list.append(coint_dict)
                pairs_counter_one_cluster += 1

            # TODO: Calculate Hurst Exponent as the second criteria for pairs selection

        print(
            f"cointegration test finished for {len(pairs_to_check)} possible pairs, "
            f"and {pairs_counter_one_cluster} pairs are found in the current cluster, ")
        pairs_counter += pairs_counter_one_cluster

    print(f"Total # of pairs found for trading day {training_date} is: {pairs_counter}")
    return coint_results_list, pd.DataFrame(true_pairs_list)


def get_pairs_for_one_trading_day(all_data, features_list, num_components, historical_days,
                                   num_training_days, significant_level, trading_date):

    # Generate features
    pca_for_the_date, explained_ratio = fea.generate_pca_features_for_clustering(all_data, num_components,
                                                                                 historical_days, features_list)

    # Asset clustering
    cluster_of_the_date = ac.Cluster(training_date=trading_date, pca_features=pca_for_the_date)
    cluster_of_the_date = cluster_of_the_date.optics()

    # Identify pairs
    conint_results, pairs_df = loop_through_cluster_list(all_data, cluster_of_the_date,
                                                         trading_date, num_training_days, significant_level)
    return pairs_df


def get_pairs_for_all_days(trading_date_beg, trading_date_end, all_data, features_list, num_components,
                               historical_days, num_training_days, significant_level=0.05):
    """ Find pairs for all the trading days within the range
        Return a dictionary where key is the trading day and value is the list of all pairs for the day
    """
    # Record run time
    start_time = datetime.datetime.now()

    # Convert trading_date_beg & trading_date_end to datetime
    beg_date = datetime.datetime.strptime(trading_date_beg, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(trading_date_end, '%Y-%m-%d')
    all_dates = all_data['date']
    date_range = all_dates[(all_dates >= beg_date) & (all_dates <= end_date)].unique()
    date_range = pd.Series(date_range).dt.strftime('%Y-%m-%d')

    # Parallelization on get_pairs_for_one_trading_day
    partial_get_pairs_for_one_day = functools.partial(get_pairs_for_one_trading_day,
                                                      all_data,
                                                      features_list,
                                                      num_components,
                                                      historical_days,
                                                      num_training_days,
                                                      significant_level)
    pairs_for_all_day_list = Parallel(n_jobs=1)(delayed(partial_get_pairs_for_one_day)(trading_date) for trading_date in date_range)
    pairs_for_all_days_df = pd.concat(pairs_for_all_day_list, axis=0)

    # Save the pairs df to csv
    save_path = Path('data/pairs_for_all_days.pkl')
    pairs_for_all_days_df.to_pickle(save_path)

    # Record run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')

    return pairs_for_all_days_df


if __name__ == '__main__':

    data_path = Path('data/cleaned_data.pkl')
    data = pd.read_pickle(data_path)

    # Generate features
    feature_list = ['return', 'cum_return', 'volume', 'current_eps', 'dividend_yield']
    pca_components = 4
    historical_days = 20
    # trade_date = datetime.datetime.strptime('2020-12-31', '%Y-%m-%d')

    # pairs_of_the_day = get_pairs_for_one_trading_day(all_data=data, features_list=feature_list, num_components=pca_components,
    #                                               historical_days=20, num_training_days=120, significant_level=0.01, trading_date=trade_date)

    # print(pairs_of_the_day)
    #
    # Get all pairs
    all_pairs = get_pairs_for_all_days(trading_date_beg='2011-01-04', trading_date_end='2020-12-31',
                                                  all_data=data, features_list=feature_list, num_components=pca_components,
                                                  historical_days=20, num_training_days=120, significant_level=0.01)
    print(all_pairs)
