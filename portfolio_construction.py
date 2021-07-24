import pandas as pd
import datetime
import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer(object):
    def __init__(self):
        pass

    def get_weights(self) -> np.array:
        pass


class EqualWeight(PortfolioOptimizer):
    def get_weights(self, asset_series: pd.Series) -> np.array:
        n = asset_series.count()
        return np.full(n, 1/n)


class MarketValueWeight(PortfolioOptimizer):
    def get_weights(self, portfolio_data: pd.DataFrame) -> np.array:
        return portfolio_data['market_cap'] / portfolio_data['market_cap'].sum()


class InverseVolatility(PortfolioOptimizer):
    def get_weights(self, volatility_series: pd.Series) -> np.array:
        volatility = volatility_series.to_numpy()
        return (1 / volatility) / sum(1 / volatility)


class EqualRiskContribution(PortfolioOptimizer):

    @staticmethod
    def calculate_covariance_matrix(portfolio_data: pd.DataFrame) -> np.array:
        # Done
        asset_returns = portfolio_data.pivot(index='date', columns='security_id', values=['daily_return'])
        asset_returns = asset_returns.to_numpy()
        return np.cov(asset_returns.T)

    def get_weights(self, covariance: np.array) -> np.array:
        """This will get the weights of each asset within the portfolio
           Note: this optimization method only solves for long only portfolio
        """
        n = len(covariance)

        def objective_function(x):
            return 1/2 * np.matmul(np.matmul(x.T, covariance), x) - 1/n * sum(np.log(x))

        optimal_x = minimize(fun=objective_function,
                             x0=np.full(n, 1 / n),
                             constraints=({'type': 'ineq', 'fun': lambda x: x}),
                             tol=1e-10,
                             ).x

        # return the normalized weights such that all weights sum to 1
        return optimal_x / sum(optimal_x)


class MeanVariance(PortfolioOptimizer):
    def get_weights(self, expected_return: np.array, covariance: np.matrix) -> np.array:
        # TODO: Need to finish this part
        pass


def find_optimal_portfolio_weights(portfolio_data: pd.DataFrame, optimization_type: str):
    """
    portfolio_data: {DataFrame} which must contain following columns:
       - date
       - security_id
       - market_cap
       - daily_return
    """
    if optimization_type == 'equally_weighted':
        optimizer = EqualWeight()
        optimal_weights = optimizer.get_weights(portfolio_data['security_id'])

    elif optimization_type == 'market_value_weighted':
        optimizer = MarketValueWeight()
        optimal_weights = optimizer.get_weights(portfolio_data['security_id'])

    elif optimization_type == 'inverse_volatility_weighted':
        optimizer = InverseVolatility()
        # Get the assets volatility series -> calculated by std of  daily return in the past half year
        volatility_series = portfolio_data.groupby('security_id')['daily_return'].apply(lambda x: x.rolling(126).std())
        optimal_weights = optimizer.get_weights(volatility_series)

    elif optimization_type == 'equal_risk_contribution':
        optimizer = EqualRiskContribution()
        covariance = optimizer.calculate_covariance_matrix(portfolio_data)
        optimal_weights = optimizer.get_weights(covariance)

    elif optimization_type == 'mean_variance':
        optimizer = MeanVariance()
        optimal_weights = optimizer.get_weights()

    # Create a new column and assign portfolio weights
    portfolio_data['weights'] = optimal_weights
