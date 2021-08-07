import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from itertools import combinations
from joblib import Parallel, delayed
import functools

import statsmodels
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

import CONFIG


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


def cointegration_engle(one_pair_of_asset, pair_testing_date, time_series1, time_series2):
    # Augmented Engle-Granger method:
    _, p_value, _, = coint(time_series1, time_series2)
    # Run OLS to get asset weights, TODO: this method seems wrong, need to investigate
    asset_weight1 = 1
    asset_weight2 = OLS(time_series1, time_series2).fit().params.values[0]

    return {'training_date': pair_testing_date,
            'asset1_gvkey': one_pair_of_asset[0],
            'asset2_gvkey': one_pair_of_asset[1],
            'asset1_weight': asset_weight1,
            'asset2_weight': asset_weight2,
            'p_value': p_value
            }


def cointegration_johansen(one_pair_of_asset, pair_testing_date, time_series_df, significance_level):
    # Johansen test:
    test_result = coint_johansen(time_series_df, det_order=0, k_ar_diff=1)
    asset_weight = test_result.evec[:, 0]
    asset_weight1 = asset_weight[0]
    asset_weight2 = asset_weight[1]

    # Get artificial p-value for pairs selection later
    critical_values = test_result.cvt[0]
    trace_stat = test_result.lr1[0]
    cv_dict = {0.1: {'cv': critical_values[0]},
               0.05: {'cv': critical_values[1]},
               0.01: {'cv': critical_values[2]},
               }
    for key in cv_dict.keys():
        cv_dict[key]['p_value'] = key - 0.0001 if trace_stat > cv_dict[key]['cv'] else key + 0.001
        cv_dict[key]['reject_null'] = True if trace_stat > cv_dict[key]['cv'] else False

    # Assign the corresponding p_value for the significance_level
    p_value = cv_dict[significance_level]['p_value']

    return {'training_date': pair_testing_date,
            'asset1_gvkey': one_pair_of_asset[0],
            'asset2_gvkey': one_pair_of_asset[1],
            'asset1_weight': asset_weight1,
            'asset2_weight': asset_weight2,
            'p_value': p_value
            }


def cointegration_test_for_one_pair(all_data, one_pair_of_asset, pair_testing_date, num_historical_days: int,
                                    significance_level: float, coint_test_method: str = 'johansen'):
    # Get the correct time range
    ending_date = pair_testing_date
    beginning_date = datetime.datetime.strftime(
        (datetime.timedelta(days=-num_historical_days) + datetime.datetime.strptime(ending_date, '%Y-%m-%d')),
        '%Y-%m-%d')

    # Select the data with the targeted date range, from beginning_date to ending_date
    data_with_targeted_date_range = all_data.loc[(all_data.date >= beginning_date) & (all_data.date <= ending_date)]

    # Get the price series for asset 1 & 2
    price_series_asset1 = data_with_targeted_date_range.loc[
        data_with_targeted_date_range.GVKEY == one_pair_of_asset[0], ['adjusted_price']].reset_index()[
        'adjusted_price'].rename('p1')
    price_series_asset2 = data_with_targeted_date_range.loc[
        data_with_targeted_date_range.GVKEY == one_pair_of_asset[1], ['adjusted_price']].reset_index()[
        'adjusted_price'].rename('p2')
    price_series = pd.concat([price_series_asset1, price_series_asset2], axis=1)

    # Cointegration test: Null is not cointegrated, if p_value is < significance_level then conintegrated
    if coint_test_method == 'engle':
        coint_dict = cointegration_engle(one_pair_of_asset, pair_testing_date, price_series['p1'], price_series['p2'])
    elif coint_test_method == 'johansen':
        coint_dict = cointegration_johansen(one_pair_of_asset, pair_testing_date, price_series[['p1', 'p2']],
                                            significance_level)
    elif coint_test_method == 'both':
        # Test on engle method first
        coint_dict = cointegration_engle(one_pair_of_asset, pair_testing_date, price_series['p1'], price_series['p2'])
        # If pass engle method, then use Johansen as second test; if doesn't pass engle, then no need to run Johansen
        if coint_dict['p_value'] < significance_level:
            coint_dict = cointegration_johansen(one_pair_of_asset, pair_testing_date, price_series[['p1', 'p2']],
                                                significance_level)

    return coint_dict


def loop_through_cluster_list(all_data, cluster_of_the_date, pair_testing_date, num_historical_days, significance_level,
                              coint_test_method):
    """ Loop through the cluster list and get all pairs for the given trading date
        Return a dictionary storing info of all possible paris --> coint_results_list
        Return a DataFrame storing all true pairs for the given training_date
    """

    possible_pairs_list = get_possible_pairs_for_each_cluster(cluster_of_the_date)
    coint_results_list = []
    true_pairs_list = []
    pairs_counter = 0

    for pairs_to_check in possible_pairs_list:
        pairs_counter_one_cluster = 0
        for i in range(0, len(pairs_to_check)):
            one_pair = pairs_to_check.iloc[i]
            coint_dict = cointegration_test_for_one_pair(all_data, one_pair, pair_testing_date, num_historical_days,
                                                         significance_level, coint_test_method)
            coint_results_list.append(coint_dict)

            # See how many pairs we have for one day -- not final selection
            p_value = coint_dict['p_value']
            if p_value < significance_level:
                true_pairs_list.append(coint_dict)
                pairs_counter_one_cluster += 1

            # TODO: Calculate Hurst Exponent as the second criteria for pairs selection, if H < 0.5 then mean reverting

        print(f"cointegration test finished for {len(pairs_to_check)} possible pairs, "
              f"and {pairs_counter_one_cluster} pairs are found in the current cluster, ")
        pairs_counter += pairs_counter_one_cluster

    print(f"Total # of pairs found for trading day {pair_testing_date} is: {pairs_counter}")
    return coint_results_list, pd.DataFrame(true_pairs_list)


def get_pairs_for_one_trading_day(all_data, cluster_data, num_historical_days, significance_level, coint_test_method,
                                  trading_date):
    # Get the cluster for the current trading day
    print(f'Generating pairs for date {trading_date}')
    cluster_of_the_date = cluster_data.loc[cluster_data.date == trading_date].copy()

    # Identify pairs
    coint_results, pairs_df = loop_through_cluster_list(all_data=all_data,
                                                        cluster_of_the_date=cluster_of_the_date,
                                                        pair_testing_date=trading_date,
                                                        num_historical_days=num_historical_days,
                                                        significance_level=significance_level,
                                                        coint_test_method=coint_test_method)
    return pairs_df


def get_pairs_for_all_days(trading_date_beg, trading_date_end, all_data, cluster_data, num_historical_days,
                           coint_test_method, significance_level=0.05):
    """ Find pairs for all the trading days within the range
        Return a dictionary where key is the trading day and value is the list of all pairs for the day
    """

    # Convert trading_date_beg & trading_date_end to datetime
    beg_date = datetime.datetime.strptime(trading_date_beg, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(trading_date_end, '%Y-%m-%d')
    all_dates = all_data['date']
    date_range = all_dates[(all_dates >= beg_date) & (all_dates <= end_date)].unique()
    date_range = pd.Series(date_range).dt.strftime('%Y-%m-%d')

    # Parallelization on get_pairs_for_one_trading_day
    partial_get_pairs_for_one_day = functools.partial(get_pairs_for_one_trading_day,
                                                      all_data,
                                                      cluster_data,
                                                      num_historical_days,
                                                      significance_level,
                                                      coint_test_method)
    pairs_for_all_day_list = Parallel(n_jobs=3)(
        delayed(partial_get_pairs_for_one_day)(trading_date) for trading_date in date_range)

    pairs_for_all_day = pd.concat(pairs_for_all_day_list, axis=0)
    pairs_for_all_day = pairs_for_all_day.sort_values(by='training_date')
    return pairs_for_all_day


if __name__ == '__main__':

    # Get cleaned data
    data_path = Path(f'data/1_cleaned_data/{CONFIG.cleaned_pkl_file_name}')
    cleaned_data = pd.read_pickle(data_path)

    # Get asset cluster for all dates data
    training_freq = 'daily'
    # training_freq = 'monthly'
    clusters_for_all_dates = pd.read_pickle(f'data/3_asset_cluster/clusters_for_all_dates_{training_freq}.pkl')

    # Get pairs data for one day
    pairs_of_the_day = get_pairs_for_one_trading_day(all_data=cleaned_data, cluster_data=clusters_for_all_dates,
                                                     num_historical_days=150, significance_level=0.05,
                                                     trading_date='2001-01-04', coint_test_method='johansen')
    print(pairs_of_the_day)

    # Get pairs for all days, run by year
    pairs_list = []
    test_method = CONFIG.coint_test_method
    for year in np.arange(2001, 2021, 1):
        print(f'Getting pairs for year {year}')
        start_time = datetime.datetime.now()

        beg = str(year) + '-01-01'
        end = str(year) + '-12-31'
        all_pairs = get_pairs_for_all_days(trading_date_beg=beg, trading_date_end=end,
                                           all_data=cleaned_data, cluster_data=clusters_for_all_dates,
                                           num_historical_days=182, significance_level=0.05,
                                           coint_test_method=test_method)
        # # Save all pairs data to pkl
        all_pairs.to_pickle(f'data/4_pairs_data/pairs_for_all_days_{year}_{test_method}.pkl')
        pairs_list.append(all_pairs)

        # Record run time
        end_time = datetime.datetime.now()
        run_time = end_time - start_time
        print(f'{run_time.seconds} seconds')

    # Concat all years' pairs data into one
    pairs_data_for_all_years = pd.concat(pairs_list, axis=1)
    pairs_data_for_all_years.to_pickle(f'data/4_pairs_data/pairs_for_all_days_{test_method}.pkl')
