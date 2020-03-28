import matplotlib.pyplot as plt
import math
import os
import datetime
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import argparse

""" This script:
        - gets an Excel file with all countries information
        - makes a pandas, filtered by date
        - computes a linear regression on the log10 figures, and predicts a few days ahead
        - plots the figures and prediction

    Todo:
        - use a logarithmic yscale
        - make a prediction for deaths
        - check usage of dates, use index
        - "with current trend, number of cases is multiplied by X every day"
        - use https://covid19api.com/#details
"""


parser = argparse.ArgumentParser()
parser.add_argument("--reload", help="reload xlsx", action="store_true")
parser.add_argument("--start_date", help="Date in format 2020-3-1", default='2020-3-1')
parser.add_argument("--country", help="Select a specific country", default='France')
parser.add_argument("--all", help="All favorite countries", action="store_true")
parser.add_argument("--days_predict", help="Number of days to predict in the future", default=10, type=int)
args = parser.parse_args()

favorite_countries = [
    "France",
    "Spain",
    "United_States_of_America",
    "United_Kingdom",
    "Italy",
    "Belgium",
    "Germany",
]

# https://www.data.gouv.fr/fr/datasets/cas-confirmes-dinfection-au-covid-19-par-region/
url_input = "https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide"
file_name_output = "COVID-19-geographic-disbtribution-worldwide.xlsx"
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

def fetch_excel():
    html_text = get_html_text(url_input)
    xlsx_url = get_xlsx_url(html_text)
    save_xlsx(xlsx_url, file_name_output)


if args.reload:
    fetch_excel()
try:
    world_info = pd.read_excel(file_name_output)
except FileNotFoundError as e:
    fetch_excel()
    world_info = pd.read_excel(file_name_output)

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

world_info['date'] = world_info['dateRep']
world_info = world_info.set_index(['date'])
world_info.sort_values(by='date')
print("Countries:", sorted(set(world_info['countriesAndTerritories'])))

def get_country_info(country):
    country_info = world_info[world_info['countriesAndTerritories'].isin([country])]
    return country_info.loc[:args.start_date]

# log10
def log10_filter(x):
    if x <= 1:
        return 0.
    else:
        return np.log10(x)

def add_country_info_log(country_info):
    country_info["casesLog"] = country_info["cases"].apply(log10_filter)
    country_info["deathsLog"] = country_info["deaths"].apply(log10_filter)
    return country_info


def add_linear_regression_log_and_prediction(country_info):
    dates_original = country_info["dateRep"] # todo: use index directly
    X = dates_original.to_numpy(dtype=np.float32).reshape(-1, 1)
    Y = country_info["casesLog"].to_numpy().reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    number_days_future = args.days_predict

    dates_extended = pd.date_range('2020-03-24', periods=number_days_future)
    ix_dates_extended = pd.DatetimeIndex(dates_original).union(pd.DatetimeIndex(dates_extended))
    country_info = country_info.reindex(ix_dates_extended)

    X_extended = country_info.index.to_numpy(dtype=np.float32).reshape(-1, 1)
    Y_pred = linear_regressor.predict(X_extended)  # make predictions
    pow_10 = lambda x: math.pow(10, x)
    prediction = pd.Series(Y_pred.ravel(), name="Prediction", index=country_info.index).apply(pow_10)
    return pd.concat([country_info, prediction], axis=1, sort=False)

# Plot
def plot_country_log(country_info_log, country):
    ax = country_info_log.reset_index().plot(x='index', y=['cases', 'deaths', 'Prediction', 'popData2018'])
    ax.set_yscale('log')
    plt.xlabel("date")
    # plt.ylabel("log_10")
    plt.title("{} - Log 10 cases/deaths".format(country))
    folder_images = "saved_images"
    plt.savefig(os.path.join(folder_images, 'img_log10_{}.png'.format(country)))

def process_plot_country(country):
    country_info = get_country_info(country)
    country_info = add_country_info_log(country_info)
    country_info = add_linear_regression_log_and_prediction(country_info)
    plot_country_log(country_info, country)

if args.all:
    countries = favorite_countries
else:
    countries = [args.country]
for country in countries:
    process_plot_country(country)
plt.show()
