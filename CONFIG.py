import pandas as pd

""" This file defines some of the parameters that are needed by back testing, model training or other files.
    This file will act as a central parameters management
"""

# 1. Original data: Params used to look up corresponding data
num_years = 20
# sectors_num = 4  # Original 2
sectors_num = 'all'

# 3. Asset Clustering
asset_cluster_training_freq = 'daily'
# asset_cluster_training_freq = 'monthly'


# 4. Identify Pairs
coint_test_method = 'both'
# coint_test_method = 'johansen'
# coint_test_method = 'engle'  # Should not use this method since the weights are not correct


# 5. Spread Feature Engineering
upper_threshold_factor = 0.8
lower_threshold_factor = 0.8

max_holding_period_days = 30
target_return = 0.03

spread_cols = ['spread_t' + str(i) for i in range(0, max_holding_period_days + 1)]
spread_daily_return_cols = ['spread_daily_return_d' + str(i) for i in range(1, max_holding_period_days + 1)]
spread_cum_return_cols = ['spread_cum_return_' + str(i) + 'd' for i in range(1, max_holding_period_days + 1)]

y_info_columns = ['prediction_date', 'evaluation_date', 'GVKEY_asset1', 'GVKEY_asset2',
                  'spread_return_60d_std'] + spread_cols + spread_daily_return_cols + spread_cum_return_cols

X_info_columns = ['GVKEY_asset1', 'GVKEY_asset2', 'prediction_date', 'evaluation_date',
                  'asset1_weight', 'asset2_weight']

# 6. Model Training
model_type = 'logistic'
# model_type = 'decision_tree'
# model_type = 'random_forest'
score_method = 'f1_macro'
param_dist = 1
random_state_num = 42
sample_weighted = True
higher_weight_factor = 1.2
weight_tag = 'with_sample_weight' if sample_weighted else 'no_sample_weight'

# 7. Trading Strategy & Back Testing
beg_date = '2020-01-01'
end_date = '2020-12-31'
prob_predicted_trade = 0.5  # if too high, issue is not enough trades
cap_weight = 0.2  # If too low, profitable trades can't have big profits
stop_loss_factor = 0.8
window_size_year = 'all'  # 5  # Specify when select historical data for model training, how long we go back, can be all

# Tag in the saved file for current run
run_version = 1
tag_for_current_run = 'triple_barrier_test_2020'

# Data Path management:

# 0. Other data
vix_original_path = f'data/0_other_data/VIX_original.csv'
vix_path = f'data/0_other_data/VIX.csv'

# 1. Original Data
cleaned_pkl_file_name = f'cleaned_data_{num_years}y_{sectors_num}sectors.pkl'
cleaned_data_path = f'data/1_cleaned_data/{cleaned_pkl_file_name}'

# 2. PCA Features
pca_features_path = f'data/2_pca_features/pca_features.pkl'

# 3. Asset Clustering
cluster_data_path = f'data/3_asset_cluster/clusters_for_all_dates_{asset_cluster_training_freq}.pkl'

# 4. Identify Pairs
pairs_data_path = f'data/4_pairs_data/{coint_test_method}/pairs_for_all_days_{coint_test_method}.pkl'

# 5. Spread Feature Engineering
spread_folder = f'data/5_spread_features/{coint_test_method}/holding_period_{max_holding_period_days}days'

pairs_label_data_path = f'{spread_folder}/pairs_label_y_target_r{target_return}.pkl'
pairs_features_data_path = f'{spread_folder}/pairs_features_X_target_r{target_return}.pkl'

# 9. BackTest Performance Results
current_run_folder = f'data/9_backtest_results/{model_type}/{coint_test_method}/v{run_version}_{tag_for_current_run}'

daily_returns_path = f'{current_run_folder}/daily_returns.csv'
monthly_returns_path = f'{current_run_folder}/monthly_returns.csv'
annual_returns_path = f'{current_run_folder}/annual_returns.csv'
performance_summary_path = f'{current_run_folder}/total_performance.csv'
return_plot_path = f'{current_run_folder}/return_plot.pdf'
portfolio_holdings_path = f'{current_run_folder}/port_holdings.csv'

# Parameters Dictionary --> info saved for analysis
current_run_info_dict = {'cointegration test method': coint_test_method,
                         'max_holding_period_days': max_holding_period_days,
                         'target_return': target_return,
                         'model_type': model_type,
                         'score_method': score_method,
                         'param_dist': param_dist,
                         'random_state_num': random_state_num,
                         'sample_weighted': sample_weighted,
                         'sample_weighted_higher_weight_factor': higher_weight_factor,
                         'backtest_beg_date': beg_date,
                         'backtest_end_date': end_date,
                         'required_prob_for_predicted_trade': prob_predicted_trade,
                         'cap_weight': cap_weight,
                         'window_size_years': window_size_year,
                         'tag_for_current_run': tag_for_current_run
                         }

current_run_info_df = pd.DataFrame.from_dict(current_run_info_dict, orient='index')
current_run_info_path = f'{current_run_folder}/current_run_info.csv'
