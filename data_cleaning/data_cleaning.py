import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


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
                         "cshoc": "shares_outstanding",
                         "cshtrd": "trading_volume",
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
                 'price_open', 'price_high', 'price_low',
                 'shares_outstanding', 'trading_volume', 'dividend_per_share', 'current_eps',
                 'sector_name', 'sector', 'industry_group', 'industry', 'sub_industry',
                 'state', 'city', 'incorp_code', 'stock_exchg_code']]

    return data


def select_time_period(data, start_date, end_date):
    """
    :param data:
    :param start_date:
    :param end_date:
    :return: return the data with the given time period
    """

    # Get the stocks that have historical data from 2010-2020 (10 years)
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # In case company name & ticker changed, group by GVKEY which is unique
    group_by_comp = data.groupby('GVKEY')
    data = group_by_comp.filter(lambda x: x['date'].min() <= start_date and x['date'].max() >= end_date)

    # Only interested in the selected time period
    data = data[data['date'] >= start_date]
    data = data[data['date'] <= end_date]

    return data


def remove_companies_with_wrong_dates(data):
    # Drop the stocks that have too many dates record or not enough date records
    date_length = data.groupby(['GVKEY'])['date'].count()
    expected_num_obs = date_length.median()
    date_obs_mask = data['GVKEY'].isin(date_length.loc[date_length == expected_num_obs].index)
    data = data.loc[date_obs_mask]

    return data


def remove_companies_with_missing_returns(data):
    # Drop the stocks that have zero record of total_return_index
    return_index_len = data.groupby(['GVKEY'])['daily_total_return_factor'].count()
    return_obs_mask = data['GVKEY'].isin(return_index_len.loc[return_index_len != 0].index)
    data = data.loc[return_obs_mask]

    return data


def clean_crsp_data(data_path: Path, output_path: Path):
    """
    Function to clean and filter CRSP data
    :param data_path: Path to original data
    :param output_path: Path to save the output data
    :return:
    """
    start_time = datetime.now()
    print("start cleaning data")

    data = pd.read_csv(data_path)

    # Convert datadate column to datetime
    data['datadate'] = pd.to_datetime(data['datadate'], format='%Y/%m/%d')

    # Sort by company & date
    data.sort_values(by=['GVKEY', 'datadate'], inplace=True)

    # Create a column: sector names
    data = map_sector_code_to_name(data)

    # Rename columns
    data = rename_columns(data)

    # Get the data within the target period:
    data = select_time_period(data, start_date='2010-01-04', end_date='2020-12-31')

    # Get financials sector stocks only
    data = data[data['sector_name'] == 'financials']

    # Drop dupes
    data.drop_duplicates(inplace=True)

    # Drop companies that have insufficient history or have duplicated dates
    data = remove_companies_with_wrong_dates(data)

    # Drop the stocks that have zero record of total_return_index
    data = remove_companies_with_missing_returns(data)

    # Save the cleaned data to pickle file -- quicker to read
    data.to_pickle(output_path)

    end_time = datetime.now()
    run_time = end_time - start_time
    print(run_time)
    print(run_time.seconds)
    print("Data are cleaned")


if __name__ == '__main__':
    data_folder = Path('../data')
    crsp_data = data_folder / Path('crsp_data.csv')
    save_path = data_folder / Path('cleaned_data.pkl')

    clean_crsp_data(crsp_data, save_path)
