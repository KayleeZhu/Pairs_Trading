

# Define model type
model = 'logistic'

# Model Training
param_dist = 1
random_state_num = 4

# Trading Strategy --> Back Test
prob_predicted_trade = 0.5  # if too high, issue is not enough trades
cap_weight = 0.2  # If too low, profitable trades can't have big profits
stop_loss_factor = 0.8
