from fetch_excel import ExcelFetcher
from fetch_apis import APIFetcher
from math_utils import smooth_curve, get_error_with_smooth
from constants import path_country_data_json
import json

""" handle several data fetchers, select the best one 
"""

def get_country_info_with_error(country_name, data_fetcher):
    """ get dataframe for the country, and the error compared to smooth curve
    """
    country_info = data_fetcher.get_country_info(country_name)
    data_name = data_fetcher.get_cases_name()
    country_info[data_name + "Smooth"] = smooth_curve(country_info[data_name])
    error = get_error_with_smooth(country_info, data_name)
    return country_info, error


class DataFetcherUtils(object):

    def __init__(self, args):
        super(DataFetcherUtils, self).__init__()
        self.data_fetcher_excel = ExcelFetcher(args, args.reload)
        self.data_fetcher_api = APIFetcher(args)
        if args.excel:
            self.data_fetcher_default = self.data_fetcher_excel
        else:
            self.data_fetcher_default = self.data_fetcher_api
        self.data_names = self.data_fetcher_default.get_data_names()
        self.all_countries = self.data_fetcher_default.get_all_countries()
        self.countries_past_predictions = {
            data_name: self.get_past_predictions_all_coutries(data_name)
            for data_name in self.data_names
        }


    def get_best_source(self, country_name):
        """ Chose best data fetcher based on error compared to smoothened curve
        """
        # todo: the errors are due to a country name not present in the lists from different sources
        try:
            country_info_excel, excel_error = \
                get_country_info_with_error(country_name, self.data_fetcher_excel)
        except (KeyError, TypeError) as e:
            print("Chose API (Excel doesn't work")
            return self.data_fetcher_api
        try:
            country_info_api, api_error = \
                get_country_info_with_error(country_name, self.data_fetcher_api)
        except (KeyError, TypeError) as e:
            print("Chose Excel (API doesn't work")
            return self.data_fetcher_excel
        if api_error < excel_error:
            print("Chose API")
            return self.data_fetcher_api
        else:
            print("Chose Excel")
            return self.data_fetcher_excel

    def get_past_predictions_all_coutries(self, data_name):
        with open(path_country_data_json, "r") as json_file:
            json_data = json.load(json_file)
        return {
            country_data["country"]: country_data["past_predictions_" + data_name]
            for country_data in json_data
        }

    def get_past_predictions(self, country_name, data_name):
        try:
            return self.countries_past_predictions[data_name][country_name]
        except KeyError as error:
            return {}
