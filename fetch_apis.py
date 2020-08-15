import requests
import pandas as pd
from pprint import pprint

from fetch_data_class import DataFetcher
from constants import *


def get_all_countries_info_by_api():
    all_countries_command = "https://corona-api.com/countries"
    response = requests.get(all_countries_command)
    obj_response = response.json()
    return obj_response["data"]


def get_country_by_api(country_code):
    country_command = "http://corona-api.com/countries/{}".format(country_code)
    response = requests.get(country_command)
    obj_response = response.json()
    country_data = obj_response["data"]
    country_timeline = country_data["timeline"]
    if not country_timeline:  # empty list
        print("{}: no country timeline (data: {})".format(country_data["name"], country_data))
        return None
    # don't take today's data
    country_timeline = country_timeline[1:]
    country_df = pd.DataFrame.from_dict(country_timeline)
    country_df["date"] = country_df["date"].apply(pd.to_datetime)
    country_df = country_df.set_index(["date"])
    return country_df


class APIFetcher(DataFetcher):
    """docstring for APIFetcher"""

    def __init__(self, args):
        super(APIFetcher, self).__init__(args)

    def make_global_data(self):
        self.all_countries_reduced_data = get_all_countries_info_by_api()
        self.country_code_dict = {
            country_reduced_data["name"]: country_reduced_data["code"]
            for country_reduced_data in self.all_countries_reduced_data
        }
        self.populations_dict = {
            country_reduced_data["name"]: country_reduced_data["population"]
            for country_reduced_data in self.all_countries_reduced_data
        }

    def get_cases_name(self):
        """ cases columns have different names based on sources (excel/api)
        """
        return "new_confirmed"

    def get_deaths_name(self):
        """ deaths columns have different names based on sources (excel/api)
        """
        return "new_deaths"

    def country_has_enough_cases(self, max_cases_country):
        return max_cases_country > min_total_cases

    def get_countries_max_cases_dict(self):
        return { # todo: also check daily cases? careful with China...
            country_reduced_data["name"]: country_reduced_data["latest_data"]["confirmed"]
            for country_reduced_data in self.all_countries_reduced_data
        }

    def fetch_country_info(self, country_name):
        return get_country_by_api(self.country_code_dict[country_name])

    def get_country_population(self, country_name):
        return self.populations_dict[country_name]

if __name__ == '__main__':
    pprint(get_country_by_api("FR"))
