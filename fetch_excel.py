import requests
import pandas as pd
from tqdm import tqdm
import os

from fetch_data_class import DataFetcher
from constants import *

folder_xlsx = "."

def get_html_text(url_input):
    # get html from url
    request = requests.get(url_input)
    return request.text

def get_xlsx_url(html_text):
    # get xlsx address by taking 1st .xlsx ocurrence
    for line in html_text.split("\n"):
    # example: <div class="media-left"><i class="media-object fa fa-download" aria-hidden="true"></i></div><div class="media-body"><a href="https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-03-21.xlsx" type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet; length=232484" title="Open file in new window" target="_blank" data-toggle="tooltip" data-placement="bottom">Download today’s data on the geographic distribution of COVID-19 cases worldwide</a><span> - EN - [XLSX-227.04 KB]</span></div>
        if "media-object" in line and ".xlsx" in line:
            return line.split("href=\"")[1].split("\" type=")[0]
    return None

def save_xlsx(xlsx_url, file_name_output):
    # download xlsx from .xlsx url
    # https://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
    response = requests.get(xlsx_url, stream=True)
    path_output = os.path.join(folder_xlsx, file_name_output)
    with open(path_output, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

def fetch_excel(url_input, file_name_output):
    html_text = get_html_text(url_input)
    xlsx_url = get_xlsx_url(html_text)
    save_xlsx(xlsx_url, file_name_output)


class ExcelFetcher(DataFetcher):
    """docstring for ExcelFetcher"""
    def __init__(self, args, reload_data):
        super(ExcelFetcher, self).__init__(args)

    def make_global_data(self):
        # https://www.data.gouv.fr/fr/datasets/cas-confirmes-dinfection-au-covid-19-par-region/
        url_input = "https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide"
        file_name_output = "COVID-19-geographic-disbtribution-worldwide.xlsx"

        if self.args.reload:
            fetch_excel(url_input, file_name_output)
        try:
            self.world_info = pd.read_excel(file_name_output)
        except FileNotFoundError as e:
            fetch_excel(url_input, file_name_output)
            self.world_info = pd.read_excel(file_name_output)

        # Excel fields are:
        #    dateRep
        #    day
        #    month
        #    year
        #    cases
        #    deaths
        #    countriesAndTerritories
        #    geoId
        #    countryterritoryCode
        #    popData2018

        # todo: put this part in excel fetch script
        self.world_info["date"] = self.world_info["dateRep"]
        self.world_info = self.world_info.set_index(["date"])
        self.world_info.sort_values(by="date")
        self.all_countries_world = set(self.world_info.countriesAndTerritories)

    def get_cases_name(self):
        """ cases columns have different names based on sources (excel/api)
        """
        return "cases"

    def get_deaths_name(self):
        """ deaths columns have different names based on sources (excel/api)
        """
        return "deaths"

    def country_has_enough_cases(self, max_cases_country):
        return max_cases_country > min_new_cases

    def get_max_cases_country(self, country_name):
        country_info = self.get_country_info(country_name)
        if country_info is None:
            return None
        else:
            return country_info["cases"].max()

    def get_countries_max_cases_dict(self):
        return {
            country_name: self.get_max_cases_country(country_name)
            for country_name in self.all_countries_world
            if self.get_max_cases_country(country_name) is not None
        }
        # countries_population_dict = dict(zip(world_info.countriesAndTerritories, world_info.popData2018))

    def fetch_country_info(self, country_name):
        return self.world_info[
            self.world_info["countriesAndTerritories"].isin([country_name])]


