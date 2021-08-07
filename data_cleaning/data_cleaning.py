import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# from .. import CONFIG


def convert_csv_to_pkl(csv_path, pkl_path):
    """
    Read the csv data under data folder, convert the csv file to pkl file and save under data folder
    """
    # csv_path = Path('../data/crsp_data.csv')
    # pkl_path = Path('../data/crsp_data.pkl')
    data = pd.read_csv(csv_path)
    data.to_pickle(pkl_path)


def map_sector_code_to_name(data):
    # map sector code to sector name based on GICS standard
    sector_conditions = [data['gsector'] == 10,
                         data['gsector'] == 15,
                         data['gsector'] == 20,
                         data['gsector'] == 25,
                         data['gsector'] == 30,
                         data['gsector'] == 35,
                         data['gsector'] == 40,
                         data['gsector'] == 45,
                         data['gsector'] == 50,
                         data['gsector'] == 55,
                         data['gsector'] == 60]

    sector_names = ['energy', 'materials', 'industrials', 'consumer_discretionary', 'consumer_staples',
                    'health_care', 'financials', 'information_technology', 'communication_services',
                    'utilities', 'real_estate']

    data['sector_name'] = np.select(sector_conditions, sector_names)

    return data


def rename_columns(data):
    # Rename some of the columns
    data.rename(columns={"datadate": "date",
                         "tic": "ticker",
                         "conm": "company_name",
                         "div": "dividend_per_share",
                         "dvi": "annual_dividend",
                         "cshoc": "shares_outstanding",
                         "cshtrd": "volume",
                         "eps": "current_eps",
                         "prccd": "price_close",
                         "prchd": "price_high",
                         "prcld": "price_low",
                         "prcod": "price_open",
                         "trfd": "daily_total_return_factor",
                         "ajexdi": "adjustment_factor",
                         "exchg": "stock_exchg_code",
                         "ggroup": "industry_group",
                         "gind": "industry",
                         "gsector": "sector",
                         "gsubind": "sub_industry",
                         "incorp": "incorp_code"
                         }, inplace=True
                )

    # rearrange data columns
    data = data[['GVKEY', 'date', 'ticker', 'company_name',
                 'daily_total_return_factor', 'adjustment_factor', 'price_close',
                 'price_open', 'price_high', 'price_low', 'shares_outstanding',
                 'volume', 'dividend_per_share', 'annual_dividend', 'current_eps',
                 'sector_name', 'sector', 'industry_group', 'industry', 'sub_industry',
                 'state', 'city', 'incorp_code', 'stock_exchg_code']]  # 'cusip'

    return data


def select_time_period(data, start_date, end_date):

    # Get the stocks that have historical data of the given time period
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # In case company name & ticker changed, group by GVKEY which is unique
    group_by_comp = data.groupby('GVKEY')
    data = group_by_comp.filter(lambda x: x['date'].min() <= start_date and x['date'].max() >= end_date)

    # Only interested in the selected time period
    data = data[data['date'] >= start_date]
    data = data[data['date'] <= end_date]

    return data


def select_sectors(data, sectors_list):

    sectors_mask = data['sector_name'].isin(sectors_list)
    data = data.loc[sectors_mask]

    return data


def remove_companies_with_wrong_dates(data):
    # Drop the stocks that have too many dates record or not enough date records
    date_length = data.groupby(['GVKEY'])['date'].count()
    expected_num_obs = date_length.median()
    date_obs_mask = data['GVKEY'].isin(date_length.loc[date_length == expected_num_obs].index)
    data = data.loc[date_obs_mask]

    return data


def remove_companies_with_missing_total_return_factors(data):
    # Drop the stocks that have zero record of daily_total_return_factor
    return_factor_len = data.groupby(['GVKEY'])['daily_total_return_factor'].count()
    return_obs_mask = data['GVKEY'].isin(return_factor_len.loc[return_factor_len != 0].index)
    data = data.loc[return_obs_mask]

    return data


def remove_companies_with_missing_volumes(data):
    # Drop the stocks that have insufficient record of volumes
    vol_len = data.groupby(['GVKEY'])['volume'].count()
    expected_num_obs = vol_len.median()
    eps_obs_mask = data['GVKEY'].isin(vol_len .loc[vol_len == expected_num_obs].index)
    data = data.loc[eps_obs_mask]

    return data


def remove_companies_with_missing_eps(data):
    # Drop the stocks that have insufficient record of current eps
    eps_len = data.groupby(['GVKEY'])['current_eps'].count()
    expected_num_obs = eps_len.median()
    eps_obs_mask = data['GVKEY'].isin(eps_len .loc[eps_len == expected_num_obs].index)
    data = data.loc[eps_obs_mask]

    return data


def remove_companies_with_missing_dividend(data):
    # Drop the stocks that have insufficient record of current eps
    div_len = data.groupby(['GVKEY'])['annual_dividend'].count()
    expected_num_obs = div_len.median()
    eps_obs_mask = data['GVKEY'].isin(div_len .loc[div_len == expected_num_obs].index)
    data = data.loc[eps_obs_mask]

    return data


def calculate_daily_returns(data):
    """
    Calculate daily return of each stock
    """
    data.sort_values(by=['GVKEY', 'date'], inplace=True)

    # Calculate adjusted price which encounter stock split & dividend
    data.eval('adjusted_price = price_close / adjustment_factor * daily_total_return_factor', inplace=True)

    # Calculate returns using adjusted price
    data['return'] = data.groupby(['GVKEY'])['adjusted_price'].pct_change(fill_method='ffill')

    # Drop the rows where return is NA
    data.dropna(subset=['return'], inplace=True)
    return data


def calculate_cumulative_returns(data):
    """
    Calculate cumulative return of each stock
    """

    data['return_factor'] = data['return'] + 1
    data['cum_return'] = data.groupby(['GVKEY'])['return_factor'].cumprod() - 1

    # Drop the rows where return is NA
    data.dropna(subset=['cum_return'], inplace=True)
    return data


def calculate_rolling_returns(data):
    # Calculate rolling returns with 5 day windows
    # TODO: This function is not working currently, need to fix
    data['roll_return'] = data.groupby(['GVKEY'])['return_factor'].rolling(5).apply(lambda x: x.prod()) - 1

    # Drop the rows where return is NA
    data.dropna(subset=['roll_return'], inplace=True)
    return data


def calculate_correlation(data):
    # Calculate the correlation matrix of the investment universe
    # TODO

    return data


def calculate_dividend_yield(data):
    """
    Calculate dividend yield of each stock
    :param data: The DataFrame we need to work on which contains annual dividend and close price
    :return: DataFrame with dividend_yield column
    """
    data.eval('dividend_yield = annual_dividend / price_close', inplace=True)
    data.dropna(subset=['dividend_yield'], inplace=True)
    return data


def clean_crsp_data(data_path: Path, output_path: Path, start_date: str, end_date: str, sectors_list: list):
    """

    :param data_path: the path of the data we want to clean. The data is in pkl format
    :param output_path: where we want to save the cleaned data
    :param start_date: The start date of our targeted time period
    :param end_date: The end date of our targeted time period
    :param sectors_list: The list of sectors we are interested in
    :return: The cleaned data
    """
    start_time = datetime.now()
    print("start cleaning data")
    data = pd.read_pickle(data_path)

    # Data Cleaning part:
    # Convert datadate column to datetime
    data['datadate'] = pd.to_datetime(data['datadate'], format='%Y/%m/%d')
    data.sort_values(by=['GVKEY', 'datadate'], inplace=True)
    # Create a column: sector names
    data = map_sector_code_to_name(data)
    data = rename_columns(data)
    data = select_time_period(data, start_date, end_date)
    data = select_sectors(data, sectors_list)
    data = remove_companies_with_wrong_dates(data)
    data = remove_companies_with_missing_total_return_factors(data)
    data = remove_companies_with_missing_volumes(data)
    data = remove_companies_with_missing_eps(data)
    data = remove_companies_with_missing_dividend(data)
    data.drop_duplicates(inplace=True)

    # Calculation:
    data = calculate_daily_returns(data)
    data = calculate_cumulative_returns(data)
    # dt = calculate_rolling_returns(data)
    data = calculate_dividend_yield(data)

    # Save the cleaned data to pickle file -- quicker to read
    data.to_pickle(output_path)

    end_time = datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
    print("Data are cleaned")


if __name__ == '__main__':

    # Parameters Control:
    num_years = 20
    data_folder = Path('../data')

    # Convert csv file to pkl file
    crsp_csv_path = data_folder / Path(f'0_crsp_data/crsp_data_{num_years}y.csv')
    crsp_pkl_path = data_folder / Path(f'0_crsp_data/crsp_data_{num_years}y.pkl')
    # convert_csv_to_pkl(csv_path=crsp_csv_path, pkl_path=crsp_pkl_path)

    # Clean data:
    date_from = '2000-01-03'
    date_to = '2020-12-31'
    sectors = ['energy', 'materials', 'industrials', 'consumer_discretionary', 'consumer_staples',
               'health_care', 'financials', 'information_technology', 'communication_services',
               'utilities', 'real_estate']
    cleaned_pkl_save_path = data_folder / Path(f'1_cleaned_data/cleaned_data_{num_years}y_allsectors.pkl')
    # cleaned_pkl_save_path = data_folder / Path(CONFIG.cleaned_pkl_file_name)

    clean_crsp_data(data_path=crsp_pkl_path, output_path=cleaned_pkl_save_path,
                    start_date=date_from, end_date=date_to, sectors_list=sectors)
