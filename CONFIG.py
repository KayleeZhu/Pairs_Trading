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
# coint_test_method = 'both'
coint_test_method = 'johansen'
# coint_test_method = 'engle'  # Should not use this method since the weights are not correct


# 5. Spread Feature Engineering
upper_threshold_factor = 0.8
lower_threshold_factor = 0.8
X_info_columns = ['GVKEY_asset1', 'GVKEY_asset2', 'prediction_date', 'evaluation_date',
                  'asset1_weight', 'asset2_weight']
y_info_columns = ['prediction_date', 'evaluation_date', 'GVKEY_asset1', 'GVKEY_asset2',
                  'spread_return_60d_std',
                  'spread_t0', 'spread_t1', 'spread_t2', 'spread_t3', 'spread_t4', 'spread_t5',
                  'spread_cum_return_3d', 'spread_cum_return_4d', 'spread_cum_return_5d',
                  'spread_daily_return_d1', 'spread_daily_return_d2', 'spread_daily_return_d3',
                  'spread_daily_return_d4', 'spread_daily_return_d5']


# 6. Model Training
model_type = 'logistic'
# model_type = 'decision_tree'
# model_type = 'random_forest'
score_method = 'f1_macro'
param_dist = 1
random_state_num = 42


# 7. Trading Strategy & Back Testing
beg_date = '2003-01-01'
end_date = '2020-12-31'
prob_predicted_trade = 0.5  # if too high, issue is not enough trades
cap_weight = 0.2  # If too low, profitable trades can't have big profits
stop_loss_factor = 0.8


# Tag in the saved file for current run
tag_for_current_run = 'new_spread_test'


# Data Path management:
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
pairs_label_data_path = f'data/5_spread_features/{coint_test_method}/pairs_label_y_u{upper_threshold_factor}_d{lower_threshold_factor}.pkl'
pairs_features_data_path = f'data/5_spread_features/{coint_test_method}/pairs_features_X_u{upper_threshold_factor}_d{lower_threshold_factor}.pkl'

# 9. BackTest Performance Results
daily_returns_path = f'data/9_backtest_results/daily_returns_{tag_for_current_run}.csv'
monthly_returns_path = f'data/9_backtest_results/monthly_returns_{tag_for_current_run}.csv'
annual_returns_path = f'data/9_backtest_results/annual_returns_{tag_for_current_run}.csv'
performance_summary_path = f'data/9_backtest_results/total_performance_{tag_for_current_run}.csv'
return_plot_path = f'data/9_backtest_results/return_plot_{tag_for_current_run}.pdf'
portfolio_holdings_path = f'data/9_backtest_results/port_holdings_{tag_for_current_run}.pdf'
