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


# class MeanVariance(PortfolioOptimizer):
#     def get_weights(self, expected_return: np.array, covariance: np.matrix) -> np.array:
#


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

    elif optimization_type == 'inverse_volatility_weighted':
        optimizer = InverseVolatility()
        # Get the assets volatility series -> calculated by std of  daily return in the past half year
        volatility_series = portfolio_data.groupby('security_id')['daily_return'].apply(lambda x: x.rolling(126).std())
        optimal_weights = optimizer.get_weights(volatility_series)

    elif optimization_type == 'equal_risk_contribution':
        optimizer = EqualRiskContribution()
        covariance = optimizer.calculate_covariance_matrix(portfolio_data)
        optimal_weights = optimizer.get_weights(covariance)

    # elif optimization_type == 'mean_variance':
    #     optimizer = MeanVariance()
    #     optimal_weights = optimizer.get_weights()

    # Create a new column and assign portfolio weights
    portfolio_data['weights'] = optimal_weights

#
# class PortfolioConstructor:
#     """ This object provides 4 types of portfolio construction methods
#         1. Equally Weighted Portfolio
#         2. Market Value Weighted Portfolio
#         3. Minimum Variance
#         4. Risk Parity
#             4.1 Naive Risk Parity (Inverse Volatility Weighted)
#             4.2 Equal Risk Contribution
#             4.3 Maximum Diversification
#     """
#
#     def __init__(self, data: pd.DataFrame):
#         """
#         data: {DataFrame} which must contain following columns:
#            - date
#            - security_id
#            - market_cap
#            - daily_return
#         """
#         self.portfolio = data.copy()
#
#     def check_portfolio_data_columns(self):
#         # TODO: check if this method works? Or come up a better way to handle the check
#         if self.portfolio.columns.isin(['date', 'security_id', 'market_cap', 'daily_return']):
#             print("Columns requirement is satisfied")
#         else:
#             print("Please include all needed columns in the portfolio DataFrame")
#
#     def equally_weighted(self, rebal_frequency='daily'):
#         if rebal_frequency == 'daily':
#             self.portfolio['weight'] = 1 / self.portfolio.groupby('date')['security_id'].count()
#
#     def market_value_weighted(self, rebal_frequency='daily'):
#         if rebal_frequency == 'daily':
#             self.portfolio['weight'] = self.portfolio['market_cap'] / self.portfolio.groupby('date')['market_cap'].sum()

    # TODO: solve the minimum_variance portfolio
    # def minimum_variance(self, rebal_frequency='daily'):

    # def inverse_volatility_weighted(self, rebal_frequency='daily'):
    #     # This method will be the same as ERC if all assets correlation are 1
    #     if rebal_frequency == 'daily':
    #         self.portfolio['volatility'] = self.portfolio.groupby('security_id')['daily_return'].apply(
    #             lambda x: x.rolling(126).std())
    #         self.portfolio['inverse_vol'] = 1 / self.portfolio['volatility']
    #         self.portfolio['weight'] = self.portfolio['inverse_vol'] / self.portfolio.groupby('date')[
    #             'inverse_vol'].sum()
    #         # Drop the intermediate column
    #         self.portfolio = self.portfolio.drop(columns='inverse_vol')
    #
    # def equal_risk_contribution(self):
    #     # Take into account when assets correlation are not 1
    #     n = self.portfolio.groupby('date')['security_id'].count()
    #     covariance = self.calculate_covariance_matrix()
    #     def object(x):
    #         1/2 * np.transpose(x) * covariance * x - 1/n * np.log(x).sum()
    #
    #     optimal_x = minimize(object, np.array([0.1])).x[0]
    #     optimal_w = optimal_x / optimal_x.sum()
    #
