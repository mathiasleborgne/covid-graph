import pandas as pd
from constants import *

""" Abstract class and utilities to fetch data from Excel, Jason APIs.
"""

def add_future_index(country_info, number_days_future):
    dates_extended = pd.DatetimeIndex(pd.date_range(country_info.index[0], periods=number_days_future))
    dates_original = pd.DatetimeIndex(country_info.index)
    ix_dates_extended = dates_original.union(dates_extended)
    return country_info.reindex(ix_dates_extended)


class DataFetcher(object):
    """ Virtual class to get data frame for each country,
        from various sources (excel, API, ...)
    """

    def __init__(self, args):
        super(DataFetcher, self).__init__()
        self.args = args
        self.start_date = args.start_date
        self.make_global_data()
        self.countries_max_cases_dict = self.get_countries_max_cases_dict()

    def make_global_data(self):
        """
        """
        raise NotImplementedError()

    def get_cases_name(self):
        """ cases columns have different names based on sources (excel/api)
        """
        raise NotImplementedError()

    def get_deaths_name(self):
        """ deaths columns have different names based on sources (excel/api)
        """
        raise NotImplementedError()

    def get_data_names(self):
        return [self.get_cases_name(), self.get_deaths_name()]

    def country_has_enough_cases(self, max_cases_country):
        raise NotImplementedError()

    def get_all_countries(self):
        countries_sorted = sorted(self.countries_max_cases_dict.items(),
                                  key=lambda item: item[1], reverse=True)
        country_min_cases = [
            country_name
            for country_name, max_cases in countries_sorted
            if self.country_has_enough_cases(max_cases)
        ]
        return country_min_cases[:max_countries_display]

    def slice_from_start_date(self, country_info):
        if self.start_date is None:
            start_date = pd.Timestamp(
                country_info[country_info[self.get_cases_name()] > min_cases_start_date]
                .index[-1])
        else:
            start_date = self.start_date
        return country_info.loc[:start_date]

    def get_countries_max_cases_dict(self):
        raise NotImplementedError()

    def fetch_country_info(self, country_name):
        raise NotImplementedError()

    def get_country_info(self, country):
        """ returns None if error/no info found
        """
        country_info = self.fetch_country_info(country)
        if country_info is None:
            return None
        country_info = country_info.loc[~country_info.index.duplicated(keep='first')]
            # remove duplicated indices in index
            # https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries
        try:
            country_info = self.slice_from_start_date(country_info)
        except IndexError as e:
            print("No data found")
            return None
        country_info = add_future_index(country_info, self.args.days_predict)
        return country_info


