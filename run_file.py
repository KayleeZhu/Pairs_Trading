import pandas as pd
import numpy as np
from pathlib import Path
import datetime

from data_cleaning import data_cleaning as clean
import asset_clustering as ac
import pca_feature_engineering as pca_fea
import identify_pairs as ip
import spread_feature_engineering as spd_fea
import model_training as mdl
import trading_strategy as ts
import back_testing as bt


if __name__ == '__main__':

    # 1. Clean data:
    data_folder = Path('../data')
    crsp_data = data_folder / Path('crsp_data.pkl')
    save_path = data_folder / Path('cleaned_data.pkl')
    date_from = '2010-01-04'
    date_to = '2020-12-31'
    sectors = ['financials', 'health_care']

    clean.clean_crsp_data(data_path=crsp_data, output_path=save_path, start_date=date_from,
                          end_date=date_to, sectors_list=sectors)


    # 2. Generate PCA features used for asset clustering
    data_path = Path('data/cleaned_data.pkl')
    dt = pd.read_pickle(data_path)
    # Parameters Control --> Get PCA features for clustering
    feature_list = ['return', 'cum_return', 'volume', 'current_eps', 'dividend_yield']
    pca, explained_ratio = pca_fea.generate_pca_features_for_clustering(data=dt, num_components=4, historical_days=20,
                                                                        features_list=feature_list)


    # 3. Asset Clustering using PCA features
    asset_cluster = ac.Cluster(training_date='2020-12-31', pca_features=pca)
    asset_cluster = asset_cluster.optics()


    # 4. Identify Pairs
    data_path = Path('data/cleaned_data.pkl')
    data = pd.read_pickle(data_path)
    # Generate features
    feature_list = ['return', 'cum_return', 'volume', 'current_eps', 'dividend_yield']
    pca_components = 4
    # Get all pairs
    all_pairs = ip.get_pairs_for_all_days(trading_date_beg='2011-01-04', trading_date_end='2020-12-31',
                                          all_data=data, features_list=feature_list, num_components=pca_components,
                                          historical_days=20, num_training_days=120, significant_level=0.01)


    # 5. Spread Feature Engineering
    # Get all_data & pairs data
    data_path = Path('cleaned_data.pkl')
    pairs_path = Path('pairs_for_all_days.pkl')
    data = spd_fea.get_crsp_data(data_path)
    pairs_data = spd_fea.get_pairs_data(pairs_path)
    spread_features = spd_fea.SpreadFeature(all_data=data, pairs=pairs_data)
    pairs_features = spread_features.generate_label_y(upper_threshold_factor=0.8, lower_threshold_factor=0.8)
    # Save data
    X = spread_features.X
    y = spread_features.y
    y_pkl_path = Path('data/pairs_label.pkl')
    X_pkl_path = Path('data/pairs_features.pkl')
    y.to_pickle(y_pkl_path)
    X.to_pickle(X_pkl_path)


    # 8. Back Testing
    features, labels = mdl.read_features_label_data()
    # Record run time
    start_time = datetime.datetime.now()

    # Define Model Type
    model = 'logistic'
    # model = 'decision_tree'
    # model = 'random_forest'

    back_test = bt.BackTest(features, labels, beg_date='2013-01-01', end_date='2020-12-31', model_type=model,
                            score_method='f1_macro')
    back_test.back_testing_given_period()
    port = back_test.portfolio.port_holdings  # --> This gives you the portfolio holdings
    port_returns = back_test.calculate_daily_returns()
    sharpe_ratio = back_test.risk_and_return()

    # Save Backtest results to the folder
    backtest_folder = Path('backtest_results')
    fig = port_returns.plot(x='effective_date', y='cum_return').get_figure()
    fig.savefig(backtest_folder / Path('cum_return_plot' + model + '.pdf'))
    port_returns.to_csv(backtest_folder / Path('daily_returns' + model + '.csv'))
    port.to_csv(backtest_folder / Path('port_holdings' + model + '.csv'))

    # Record run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')