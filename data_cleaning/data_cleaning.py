import pandas as pd
import numpy as np
from pathlib import Path


def clean_crsp_data(data_path: Path, output_path: Path):
    """
    Function to clean and filter CRSP data
    :param data_path: Path to original data
    :param output_path: Path to save the output data
    :return:
    """
    data = pd.read_csv(data_path)

    data['date'] = pd.to_datetime(data['datadate'], format='%Y%m%d')

    data = data.rename(columns={
        'tic': 'ticker',
        'conm': 'company_name',
        'cshoc': 'shares_outstanding',
        'cshtrd': 'trading_volume',
        'dvi': 'current_annual_div',
        'prccd': 'price_close',
        'prchd': 'price_high',
        'prcld': 'price_low',
        'prcod': 'price_open',
        'trfd': 'total_return_index',
        'conml': 'comp_legal_name',
        'ggroup': 'industry_group',
        'gind': 'industry',
        'gsubind': 'sub_industry'}
    )

    # rearrange data columns
    data = data[['GVKEY', 'iid', 'date', 'ticker', 'cusip', 'company_name',
                 'shares_outstanding', 'trading_volume', 'current_annual_div',
                 'price_close', 'price_high', 'price_low', 'price_open',
                 'total_return_index', 'busdesc', 'comp_legal_name', 'county',
                 'industry_group', 'industry', 'sub_industry']]

    # create gics sector
    data['sector'] = data['industry_group'].astype(str).str[0:2]

    sector_conditions = [data['sector'] == '10',
                         data['sector'] == '15',
                         data['sector'] == '20',
                         data['sector'] == '25',
                         data['sector'] == '30',
                         data['sector'] == '35',
                         data['sector'] == '40',
                         data['sector'] == '45',
                         data['sector'] == '50',
                         data['sector'] == '55',
                         data['sector'] == '60']

    sector_names = ['energy', 'materials', 'industrials', 'consumer_discretionary', 'consumer_staples',
                    'health_care', 'financials', 'information_technology', 'communication_services',
                    'utilities', 'real_estate']

    data['sector_name'] = np.select(sector_conditions, sector_names)

    pd.to_pickle(output_path)


if __name__ == '__main__':
    data_folder = Path('../data')

    all_data = data_folder / Path('all_data.csv')
    save_path = data_folder / Path('cleaned_data.pkl')
    clean_crsp_data(all_data, save_path)
