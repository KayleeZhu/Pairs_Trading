import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def convert_csv_to_pkl():
    """
    Read the csv data under data folder, convert the csv file to pkl file and save under data folder
    """
    csv_path = Path('../data/crsp_data.csv')
    data = pd.read_csv(csv_path)

    pkl_path = Path('../data/crsp_data.pkl')
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
    data = data[['GVKEY', 'date', 'ticker', 'cusip', 'company_name',
                 'daily_total_return_factor', 'adjustment_factor', 'price_close',
                 'price_open', 'price_high', 'price_low', 'shares_outstanding',
                 'volume', 'dividend_per_share', 'annual_dividend', 'current_eps',
                 'sector_name', 'sector', 'industry_group', 'industry', 'sub_industry',
                 'state', 'city', 'incorp_code', 'stock_exchg_code']]

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

    # Convert datadate column to datetime
    data['datadate'] = pd.to_datetime(data['datadate'], format='%Y/%m/%d')

    # Sort by company & date
    data.sort_values(by=['GVKEY', 'datadate'], inplace=True)

    # Create a column: sector names
    data = map_sector_code_to_name(data)

    # Rename columns
    data = rename_columns(data)

    # Get the data within the target period:
    data = select_time_period(data, start_date, end_date)

    # Get the targeted sectors stocks
    data = select_sectors(data, sectors_list)

    # Drop companies that have insufficient history or have duplicated dates
    data = remove_companies_with_wrong_dates(data)

    # Drop the stocks that have zero record of total_return_factor
    data = remove_companies_with_missing_total_return_factors(data)

    # Drop the stocks that have insufficient record of volume
    data = remove_companies_with_missing_volumes(data)

    # Drop the stocks that have insufficient record of current eps
    data = remove_companies_with_missing_eps(data)

    # Drop the stocks that have insufficient record of annual dividend
    data = remove_companies_with_missing_dividend(data)

    data.drop_duplicates(inplace=True)

    # Save the cleaned data to pickle file -- quicker to read
    data.to_pickle(output_path)

    end_time = datetime.now()
    run_time = end_time - start_time
    print(f'{run_time.seconds} seconds')
    print("Data are cleaned")


if __name__ == '__main__':
    # convert_csv_to_pkl()

    # Parameters Control:
    data_folder = Path('../data')
    crsp_data = data_folder / Path('crsp_data.pkl')
    save_path = data_folder / Path('cleaned_data_4_sectors.pkl')
    date_from = '2010-01-04'
    date_to = '2020-12-31'
    sectors = ['financials', 'health_care', 'information_technology', 'communication_services']

    clean_crsp_data(data_path=crsp_data, output_path=save_path, start_date=date_from,
                    end_date=date_to, sectors_list=sectors)
