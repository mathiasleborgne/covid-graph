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
import numpy as np
from pprint import pprint

from utils import get_args, save_json
from math_utils import smooth_max, smooth_curve, get_error_with_smooth
from publish import push_if_outdated, get_today_date_str, get_date_last_update
from prediction_tools import regress_predict_data, get_column_name_func
from constants import *
from data_fetcher_utils import DataFetcherUtils

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
        - sort by max cases ever?
        - start at 50 cases
        - collapse explanation
        - french regions?
        - detect new_cases = delay1(deaths) + delay2(recovered)
        - more information from web API
        - expand country/country specific page
        - show peak
        - anchor links
        - facebook likes count
        - refactor
"""
args = get_args()
former_date = get_date_last_update()
data_fetcher_utils = DataFetcherUtils(args)
print("Countries:", data_fetcher_utils.all_countries)

favorite_countries = [
    "France",
    "United_States_of_America" if args.excel else "USA",
    "United_Kingdom" if args.excel else "UK",
    "Italy",
    "Belgium",
    "Germany",
    "Spain",
]



def get_latest_date_index(country_info, data_name, is_extended=False):
    """ get the latest date of index, NaN included if extended (=with prediction) 
    """
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
    if log_scale:
        ax.set_yscale("log")
    plt.xlabel("date")
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


def add_past_prediction(past_predictions, latest_prediction, date_timestamp):
    past_predictions[date_timestamp.strftime('%Y-%m-%d')] = latest_prediction
    return past_predictions


def export_data(country_info, data_name, country_population, smoothen=True):
    if smoothen:
        floats_array = smooth_curve(country_info[data_name].values)
    else:
        floats_array = country_info[data_name].values
    # normalizing in cases per million
    floats_array = floats_array / (country_population / 1.0e6)
    # decimals: rounding is done to lighten the JSON file,
    # in order to make website loading faster
    return floats_array.round(decimals=1).tolist()

def export_data_prediction(country_info, country_all_results, data_name,
                           country_population):
    column_name = get_column_name_func(
        data_name, country_all_results[data_name]["prediction_type"], False, True)
    return export_data(country_info, column_name, country_population, False)

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
    return country_info, country_all_results

def make_country_json_dict(country_name, country_info, data_fetcher, country_all_results):
    """ Build dict per country for subsequent JSON export
    """
    index_str_list = [str(timestamp) for timestamp in country_info.index.tolist()]
    latest_date_index = get_latest_date_index(country_info, data_fetcher.get_cases_name(), True)
    country_population = data_fetcher.get_country_population(country_name)
    prediction_confirmed = export_data_prediction(country_info, country_all_results, data_fetcher.get_cases_name(), country_population)
    prediction_deaths = export_data_prediction(country_info, country_all_results, data_fetcher.get_deaths_name(), country_population)
    country_json_dict = {
        "country_data": country_all_results,
        "country": improve_country_name(country_name),
        "dates": index_str_list,
        "new_confirmed": export_data(country_info, data_fetcher.get_cases_name(), country_population),
        "new_deaths": export_data(country_info, data_fetcher.get_deaths_name(), country_population),
        "prediction_confirmed": prediction_confirmed,
        "prediction_deaths": prediction_deaths,
        "past_predictions_new_confirmed": add_past_prediction(
            data_fetcher_utils.get_past_predictions(country_name, "new_confirmed"),
            prediction_confirmed[-1], latest_date_index),
        "past_predictions_new_deaths": add_past_prediction(
            data_fetcher_utils.get_past_predictions(country_name, "new_deaths"),
            prediction_deaths[-1], latest_date_index),
    }
    return country_json_dict

def get_countries():
    """ Get the list of countries to process based on args
    """
    if args.favorite:
        return favorite_countries
    elif args.all:
        not_favorite_countries = set(data_fetcher_utils.all_countries) - set(favorite_countries)
        return favorite_countries + list(not_favorite_countries)
    else:
        return [args.country]

def predict_all_countries(countries):
    """ Make processing/prediction for a list of countries,
        return JSON dict with countries data
    """
    images_info = []
    for index, country_name in enumerate(countries):
        try:
            print("Processing {} - ({}/{})"
                  .format(country_name, index + 1, len(countries)))
            data_fetcher_best = data_fetcher_utils.get_best_source(country_name)
            country_info = data_fetcher_best.get_country_info(country_name)
            if country_info is None:
                print("No case found for {}".format(country_name))
                continue
            print("    {} cases"
                  .format(data_fetcher_best.countries_max_cases_dict[country_name]))
            country_info, country_all_results = \
                process_plot_country(country_name, country_info,
                                     data_fetcher_best)
            country_json_dict = make_country_json_dict(country_name, country_info,
                                                       data_fetcher_best,
                                                       country_all_results)
            images_info.append(country_json_dict)
        except (ValueError, IndexError) as error:
            print("No case found for {} (error: {})".format(country_name, error))
            continue
        print()
    return images_info

def make_global_info():
    """ Make JSON dictionary with global information (not country-specific)
    """
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
