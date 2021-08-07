import pandas as pd
import datetime
import numpy as np
from pathlib import Path

import CONFIG


class Performance:

    def __init__(self, port_holdings):
        self.port_holdings = port_holdings
        self.port_daily_returns = self.calculate_daily_returns()

    def calculate_daily_returns(self):
        port = self.port_holdings.copy()
        port_daily_returns = (port.groupby('effective_date')[['contribution']]
                              .sum()
                              .reset_index()
                              .rename(columns={'contribution': 'daily_return'})
                              )
        port_daily_returns['cum_return'] = (port_daily_returns['daily_return'] + 1).cumprod() - 1

        return port_daily_returns

    def calculate_monthly_returns(self):
        port = self.port_daily_returns.copy()
        port['year'] = port['effective_date'].dt.year
        port['month'] = port['effective_date'].dt.month
        port_monthly_returns = port.groupby(['year', 'month'])['daily_return'].apply(
            lambda x: (1 + x).prod() - 1).reset_index().rename(columns={'daily_return': 'monthly_return'})

        return port_monthly_returns

    def calculate_annual_returns(self):
        port = self.port_daily_returns.copy()
        port['year'] = port['effective_date'].dt.year

        port_annual_returns = port.groupby('year')['daily_return'].apply(
            lambda x: (1 + x).prod() - 1).reset_index().rename(
            columns={'daily_return': 'annual_return'})
        annual_std = port.groupby('year')['daily_return'].apply(
            lambda x: x.std() * np.sqrt(x.count())).reset_index().rename(columns={'daily_return': 'annual_std'})
        # TODO: think about if we should use sqrt(252) to replace,
        #       the current logic will overstate SR if some days have no holdings

        port_annual_returns = port_annual_returns.merge(annual_std, on='year')
        port_annual_returns['sharpe'] = port_annual_returns['annual_return'] / port_annual_returns['annual_std']

        return port_annual_returns

    def annualized_risk_and_return(self):
        port_daily_returns = self.port_daily_returns.copy()
        total_return = (1 + port_daily_returns['daily_return']).prod() - 1
        num_years = port_daily_returns['effective_date'].max().year - port_daily_returns[
            'effective_date'].min().year + 1

        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        standard_deviation = port_daily_returns['daily_return'].std() * np.sqrt(252)
        sharpe = annualized_return / standard_deviation

        perform_dict = {'annualized_return': annualized_return,
                        'annualized_std': standard_deviation,
                        'annualized_sharpe': sharpe
                        }
        performance = pd.Series(perform_dict)

        print(f"annualized_return is {annualized_return}")
        print(f"std is {standard_deviation}")
        print(f"sharpe is {sharpe}")
        return performance

    def calculate_ytd_returns(self):
        pass


if __name__ == '__main__':
    # Record run time
    start_time = datetime.datetime.now()

    # Record run time
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
