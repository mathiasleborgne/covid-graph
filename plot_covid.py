""" Main script:
    - choose best data fetcher,
    - fit data curves,
    - make predictions,
    - put all information for website in JSON outputs
"""

import matplotlib.pyplot as plt
import json
import math
import os
import datetime
import pandas as pd
import argparse
from pprint import pprint

from fetch_excel import ExcelFetcher
from fetch_apis import APIFetcher
from math_utils import smooth_max, smooth_curve, mean_absolute_error_norm
from publish import push_if_outdated, get_today_date_str, get_date_last_update
from prediction_tools import regress_predict_data, get_column_name_func
from constants import *


""" This script:
        - gets a JSON/Excel file with all countries information
        - makes a pandas dataframe per country, filtered by date
        - computes a curve fit on the figures with
            - exponential model
            - logistics model
            - logistics then exponential model (piecewise)
        - chooses best model and predicts a few days ahead
        - plots the figures and prediction

    Todo:
        - check figures for duplicated dates
        - sort by max cases ever?
        - start at 50 cases
        - spinners
        - collapse explanation
        - french regions?
        - detect new_cases = delay1(deaths) + delay2(recovered)
        - more information from web API
        - expand country/country specific page
        - show peak
        - anchor links
        - facebook likes count
        - refactor
            - fully split maths/excel fetch
            - objects
"""

# parser -------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--reload", help="reload xlsx", action="store_true")
parser.add_argument("--start_date", help="Date in format 2020-3-1", default=None)
parser.add_argument("--country", help="Select a specific country", default="France")
parser.add_argument("--favorite", help="Favorite countries", action="store_true")
parser.add_argument("--all", help="All countries", action="store_true")
parser.add_argument("--show", help="Show images", action="store_true")
parser.add_argument("--excel", help="Get data from excel instead of api", action="store_true")
parser.add_argument("--save_imgs", help="Save images", action="store_true")
parser.add_argument("--temp_curves", help="Show temporary curves", action="store_true")
parser.add_argument("--publish", help="Commit data update, don't push", action="store_true")
parser.add_argument("--publish_push", help="Publish data update on website", action="store_true")

parser.add_argument("--days_predict", help="Number of days to predict in the future", default=number_days_future_default, type=int)
args = parser.parse_args()

path_country_data_json = os.path.join("docs", "_data", "images_info.json")
former_date = get_date_last_update()

favorite_countries = [
    "France",
    "United_States_of_America" if args.excel else "USA",
    "United_Kingdom" if args.excel else "UK",
    "Italy",
    "Belgium",
    "Germany",
    "Spain",
]

improved_country_names = {
    # when you get a bad name like those on the left,
    # replace by bame on the right
    "USA": "United States of America",
    "UK": "United kingdom",
    "S. Korea": "South Korea",
}

data_fetcher_excel = ExcelFetcher(args, args.reload)
data_fetcher_api = APIFetcher(args)
if args.excel:
    data_fetcher_default = data_fetcher_excel
else:
    data_fetcher_default = data_fetcher_api

data_names = data_fetcher_default.get_data_names()

def get_past_predictions_all_coutries(data_name):
    with open(path_country_data_json, "r") as json_file:
        json_data = json.load(json_file)
    return {
        country_data["country"]: country_data["past_predictions_" + data_name]
        for country_data in json_data
    }

countries_predictions = {
    data_name: get_past_predictions_all_coutries(data_name)
    for data_name in data_names
}

all_countries = data_fetcher_default.get_all_countries()
print("Countries:", all_countries)

def get_error_with_smooth(country_info, data_name):
    """ mean absolute error between raw and smoothened curves
    """
    country_data_smooth = country_info[data_name + "Smooth"].dropna(how="any")
    len_smooth = len(country_data_smooth)
    country_data = country_info[data_name].dropna(how="any")[:len_smooth]
    return mean_absolute_error_norm(country_data_smooth, country_data)


def get_country_info_with_error(country_name, data_fetcher):
    """ get dataframe for the country, and the error compared to smooth curve
    """
    country_info = data_fetcher.get_country_info(country_name)
    data_name = data_fetcher.get_cases_name()
    country_info[data_name + "Smooth"] = smooth_curve(country_info[data_name])
    error = get_error_with_smooth(country_info, data_name)
    return country_info, error


def get_best_source(country_name):
    """ Chose best data fetcher based on error compared to smoothened curve
    """
    # todo: the errors are due to a country name not present in the lists from different sources
    try:
        country_info_excel, excel_error = \
            get_country_info_with_error(country_name, data_fetcher_excel)
    except (KeyError, TypeError) as e:
        print("Chose API (Excel doesn't work")
        return data_fetcher_api
    try:
        country_info_api, api_error = \
            get_country_info_with_error(country_name, data_fetcher_api)
    except (KeyError, TypeError) as e:
        print("Chose Excel (API doesn't work")
        return data_fetcher_excel
    if api_error < excel_error:
        print("Chose API")
        return data_fetcher_api
    else:
        print("Chose Excel")
        return data_fetcher_excel

def get_latest_date_index(country_info, data_name, is_extended=False):
    if is_extended:
        return country_info[data_name].index[-1]
    else:
        return country_info[data_name].dropna(how="any").index[-1]

def get_peak_date(country_info, data_name):
    """ return peak date as pandas timestamp if detected, else None
    """
    max_country, argmax_country = smooth_max(country_info, data_name)
    country_data_smooth = country_info[data_name + "Smooth"].dropna(how="any")
    today = get_latest_date_index(country_info, data_name)
    days_after_peak = (today - argmax_country).days
    latest_value = country_data_smooth[-1]
    if max_country == 0.:
        return None
    decrease_pct = float(max_country - latest_value) / max_country * 100.
    # todo: decrease condition on mean after peak

    peak_detected = days_after_peak >= min_days_post_peak \
        and decrease_pct > min_decrease_post_peak
    if peak_detected:
        return pd.Timestamp(argmax_country)
    else:
        return None

# Plot ----------------------------------
def plot_country_log(country, all_results, country_info, data_fetcher, log_scale):
    """ Plot country data and predictions, with log scale (or not, based on log_scale argument).
        To be used locally, for the website we use JSON export + JS lib
    """
    data_names = data_fetcher.get_data_names()
    prediction_columns_names = [
        get_column_name_func(data_name, all_results[data_name]["prediction_type"], False, True)
        for data_name in data_names
    ]
    if args.temp_curves:
        prediction_columns_names += ["casesSmooth", "deathsSmooth"]
    ax = country_info.reset_index()\
        .plot(x="index", y=data_names + prediction_columns_names)
    # ax = country_info.reset_index().plot(x="index", y=["casesLog", "deathsLog", "PredictionLog"])
    # ax = country_info.reset_index().plot(x="index", y=["cases", "deaths"])
    if log_scale:
        ax.set_yscale("log")
    plt.xlabel("date")

    # plt.ylabel("log_10")
    if args.show:
        case_data_name = data_names[0]
        plt.title("{} - Cases/Deaths\n(Reg. error: {:.1f} pct / Daily growth: {:.1f} pct)"
                  .format(country, all_results[case_data_name]["reg_error_pct"],
                          all_results[case_data_name]["daily_growth_pct"]))
    folder_images = "saved_images"
    image_name = "img_log10_{}_{}.png".format(country, "log" if log_scale else "normal")
    if args.save_imgs:
        plt.savefig(os.path.join(folder_images, image_name))
        plt.savefig(os.path.join("docs", "assets", "img", image_name))
    return image_name

def improve_country_name(country_name):
    """ Get a better country name if possible, based on hardcoded dict, else just return the country name
    """
    try:
        return improved_country_names[country_name]
    except KeyError as e:
        return country_name

def get_past_predictions(country_name, data_name):
    try:
        return countries_predictions[data_name][country_name]
    except KeyError as error:
        return {}

def add_prediction(past_predictions, latest_prediction, date_timestamp):
    past_predictions[date_timestamp.strftime('%Y-%m-%d')] = latest_prediction
    return past_predictions

def process_plot_country(country_name, country_info, data_fetcher):
    """ Get all data (including predictions) for a country
        Return a dict with the results, for later JSON conversion
    """
    country_all_results = {}

    for data_name in data_fetcher.get_data_names():
        print("Analyzing {}".format(data_name))
        country_info[data_name + "Smooth"] = smooth_curve(country_info[data_name])
        error_with_smooth = get_error_with_smooth(country_info, data_name)
        print("error_with_smooth:", error_with_smooth)
        peak_date = get_peak_date(country_info, data_name) # needs smooth column addition...
        is_peak = peak_date is not None
        print("    is peak? {} - date max: {}".format(is_peak, peak_date))
        start_date = pd.Timestamp(country_info.index[0])
        prediction_date = pd.Timestamp(country_info.index[-1])
            # todo: not same type...
        updated_country_info, country_results_data = \
            regress_predict_data(data_name, country_info, is_peak)
        country_all_results[data_name] = country_results_data
        country_info = updated_country_info

    image_name_log = plot_country_log(country_name, country_all_results, country_info, data_fetcher, True)
    image_name_normal = plot_country_log(country_name, country_all_results, country_info, data_fetcher, False)
    index_str_list = [str(timestamp) for timestamp in country_info.index.tolist()]

    def export_data(data_name, smoothen=True):
        if smoothen:
            return smooth_curve(country_info[data_name].values).tolist()
        else:
            return country_info[data_name].values.tolist()

    def export_data_prediction(data_name):
        column_name = get_column_name_func(
            data_name, country_all_results[data_name]["prediction_type"], False, True)
        return export_data(column_name, False)
    latest_date_index = get_latest_date_index(country_info, data_fetcher.get_cases_name(), True)
    country_all_results = {
        "country_data": country_all_results,
        "country": improve_country_name(country_name),
        "image_name_log": image_name_log,
        "image_name_normal": image_name_normal,
        "dates": index_str_list,
        "new_confirmed": export_data(data_fetcher.get_cases_name()),
        "new_deaths": export_data(data_fetcher.get_deaths_name()),
        "prediction_confirmed": export_data_prediction(data_fetcher.get_cases_name()),
        "prediction_deaths": export_data_prediction(data_fetcher.get_deaths_name()),
        "past_predictions_new_confirmed": add_prediction(
            get_past_predictions(country_name, "new_confirmed"),
            export_data_prediction(data_fetcher.get_cases_name())[-1],
            latest_date_index),
        "past_predictions_new_deaths": add_prediction(
            get_past_predictions(country_name, "new_deaths"),
            export_data_prediction(data_fetcher.get_deaths_name())[-1],
            latest_date_index),
    }
    return country_all_results

def save_json(file_name, content):
    """ Save as JSON file
    """
    with open(file_name, "w") as outfile:
        json.dump(content, outfile, indent=4)

def get_countries():
    """ Get the list of countries to process based on args
    """
    if args.favorite:
        return favorite_countries
    elif args.all:
        not_favorite_countries = set(all_countries) - set(favorite_countries)
        return favorite_countries + list(not_favorite_countries)
    else:
        return [args.country]

def predict_all_countries(countries):
    images_info = []
    for index, country_name in enumerate(countries):
        try:
            print("Processing {} - ({}/{})"
                  .format(country_name, index + 1, len(countries)))
            data_fetcher_best = get_best_source(country_name)
            country_info = data_fetcher_best.get_country_info(country_name)
            if country_info is None:
                print("No case found for {}".format(country_name))
                continue
            print("    {} cases"
                  .format(data_fetcher_best.countries_max_cases_dict[country_name]))
            image_info = process_plot_country(country_name, country_info, data_fetcher_best)
            images_info.append(image_info)
        except (ValueError, IndexError) as error:
            print("No case found for {} (error: {})".format(country_name, error))
            continue
        print()
    return images_info

def make_global_info():
    return {
        "days_predict": args.days_predict,
        "favorite_countries": favorite_countries,
        "min_new_cases": min_new_cases,
        "min_total_cases": min_total_cases,
        "date_last_update": get_today_date_str(),
    }


countries = get_countries()
images_info = predict_all_countries(countries)
save_json(path_country_data_json, images_info)
save_json(os.path.join("docs", "_data", "global_info.json"), make_global_info())

if args.show and images_info:
    plt.show()

if args.publish or args.publish_push:
    push_if_outdated(args.publish_push, former_date)
