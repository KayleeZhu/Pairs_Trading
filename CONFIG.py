
""" This file defines some of the parameters that are needed by back testing, model training or other files.
    This file will act as a central parameters management
"""


# Used to look up corresponding data
num_years = 20
# sectors_num = 4  # Original 2
sectors_num = 'all'
cleaned_pkl_file_name = f'cleaned_data_{num_years}y_{sectors_num}sectors.pkl'

# Identify Pairs
coint_test_method = 'both'
# coint_test_method = 'johansen'


# Define model type
model = 'logistic'
# model = 'decision_tree'
# model = 'random_forest'


# Model Training
param_dist = 1
random_state_num = 4


# Trading Strategy --> Back Test
prob_predicted_trade = 0.5  # if too high, issue is not enough trades
cap_weight = 0.2  # If too low, profitable trades can't have big profits
stop_loss_factor = 0.8


# Tag in the saved file for current run
tag_for_current_run = 'original_no_stop_loss'
