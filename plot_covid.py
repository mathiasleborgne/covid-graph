import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import json
import math
import os
import datetime
import numpy as np
import scipy
import pandas as pd
import argparse
from pprint import pprint

from fetch_excel import fetch_excel
from fetch_apis import get_country_by_api, get_all_countries_info_by_api
from math_utils import smooth_max, get_applied_func, smooth_curve, get_float_index
from publish import push_if_outdated, get_today_date_str, get_date_last_update


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
# constants --------------------------------------
min_new_cases = 100
min_total_cases = 1000
min_cases_start_date = 30
min_days_post_peak = 8
min_decrease_post_peak = 10.
number_days_future_default = 10
max_countries_display = 50  # max number of countries to display

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


def get_cases_name():
    """ cases columns have different names based on sources (excel/api)
    """
    return "new_confirmed" if not args.excel else "cases"

def get_deaths_name():
    """ deaths columns have different names based on sources (excel/api)
    """
    return "new_deaths" if not args.excel else "deaths"

former_date = get_date_last_update()
data_names = [get_cases_name(), get_deaths_name()]

favorite_countries = [
    "France",
    "Spain",
    "United_States_of_America" if args.excel else "USA",
    "United_Kingdom" if args.excel else "UK",
    "Italy",
    "Belgium",
    "Germany",
]

improved_country_names = {
    # when you get a bad name like those on the left,
    # replace by bame on the right
    "USA": "United States of America",
    "UK": "United kingdom",
    "S. Korea": "South Korea",
}

# https://www.data.gouv.fr/fr/datasets/cas-confirmes-dinfection-au-covid-19-par-region/
url_input = "https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide"
file_name_output = "COVID-19-geographic-disbtribution-worldwide.xlsx"

if args.reload:
    fetch_excel(url_input, file_name_output)
try:
    world_info = pd.read_excel(file_name_output)
except FileNotFoundError as e:
    fetch_excel(url_input, file_name_output)
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

# todo: put this part in excel fetch script
world_info["date"] = world_info["dateRep"]
world_info = world_info.set_index(["date"])
world_info.sort_values(by="date")
all_countries_world = set(world_info.countriesAndTerritories)


def slice_from_start_date(country_info):
    if args.start_date is None:
        start_date = pd.Timestamp(
            country_info[country_info[get_cases_name()] > min_cases_start_date]
            .index[-1])
    else:
        start_date = args.start_date
    return country_info.loc[:start_date]

def add_future_index(country_info, number_days_future):
    dates_extended = pd.DatetimeIndex(pd.date_range(country_info.index[0], periods=number_days_future))
    dates_original = pd.DatetimeIndex(country_info.index)
    ix_dates_extended = dates_original.union(dates_extended)
    return country_info.reindex(ix_dates_extended)

def get_country_info(country, force_excel=False):
    if args.excel or force_excel:
        country_info = world_info[world_info["countriesAndTerritories"].isin([country])]
    else:
        country_info = get_country_by_api(country_code_dict[country])
    if country_info is None:
        return None
    country_info = country_info.loc[~country_info.index.duplicated(keep='first')]
        # remove duplicated indices in index
        # https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries
    country_info = slice_from_start_date(country_info)
    country_info = add_future_index(country_info, args.days_predict)
    return country_info

if args.excel:
    countries_max_cases_dict = {
        country: get_country_info(country, force_excel=True)["cases"].max()
        for country in all_countries_world
    }
    # countries_population_dict = dict(zip(world_info.countriesAndTerritories, world_info.popData2018))
else:
    all_countries_reduced_data = get_all_countries_info_by_api()
    country_code_dict = {
        country_reduced_data["name"]: country_reduced_data["code"]
        for country_reduced_data in all_countries_reduced_data
    }
    countries_max_cases_dict = { # todo: also check daily cases? careful with China...
        country_reduced_data["name"]: country_reduced_data["latest_data"]["confirmed"]
        for country_reduced_data in all_countries_reduced_data
    }

countries_sorted = sorted(countries_max_cases_dict.items(),
                          key=lambda item: item[1], reverse=True)
country_min_cases = [
    country_name
    for country_name, max_cases in countries_sorted
    if max_cases > (min_new_cases if args.excel else min_total_cases)
]
all_countries = country_min_cases[:max_countries_display]
print("Countries:", all_countries)


def get_peak_date(country_info, data_name):
    max_country, argmax_country = smooth_max(country_info, data_name)
    country_data_smooth = country_info[data_name + "Smooth"].dropna(how="any")
    today = country_info[data_name].dropna(how="any").index[-1]
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

def get_column_name_func(column_name, prediction_type, is_inverted, is_prediction):
    # todo: inverted and prediction: compatible?
    column_name_func = column_name + prediction_type
    if is_prediction:
        column_name_func = column_name_func + "Prediction"
    if is_inverted:
        column_name_func = column_name_func + "Inv"
    return column_name_func

# regression -----------------------------
def add_linear_regression_log_and_prediction(
    country_info, data_name, applied_func, prediction_type):
    """ use scipy's curve_fit to fit any function
    """
    column_applied_func = get_column_name_func(data_name, prediction_type, True, False)
    # start_date, end_date = date_range
    # country_info_ranged = country_info.loc[start_date:end_date]
    # country_info_filtered = country_info.loc[start_date:end_date].dropna(how="any")
    country_data = country_info[data_name]
    country_data_ranged = country_data # todo: useful to add range?
    country_data_filtered = country_data.dropna(how="any")
    X = country_data_filtered.index.to_numpy(dtype=np.float32).reshape(-1, 1).ravel()
    X_extended = get_float_index(country_data_ranged)
        # todo back to timestamps?
    X = X_extended[:len(country_data_filtered.index)]
    Y = country_data_filtered.to_numpy().reshape(-1, 1).ravel()

    popt, pcov = scipy.optimize.curve_fit(applied_func, X, Y)
    applied_func_params = lambda x: applied_func(x, *popt)

    reg_error_pct = mean_absolute_error(Y, applied_func_params(X))/ np.mean(Y) * 100

    # predict
    # X_extended = country_data_ranged.index.to_numpy(dtype=np.float32).reshape(-1, 1).ravel()
    Y_pred = applied_func_params(X_extended)

    # compute daily multiplicative factor
    # eg for Exponential:
    # ln(Y) = a*t+b --> Y(t) = B*exp(a*t) --> Y(t+1) = B*exp(a)*exp(a*t) = exp(a)*Y(t)
    daily_growth_pct = (applied_func_params(X[-1]) - applied_func_params(X[-2]))/applied_func_params(X[-1]) * 100
    # print("{} grow of {} pct each day".format(data_name, daily_growth_pct))

    # add to dataframe
    prediction = pd.Series(np.round(Y_pred).ravel(),
                           name=get_column_name_func(data_name, prediction_type, False, True),
                           index=country_data_ranged.index)
    country_info = pd.concat([country_info, prediction],
                              axis=1, sort=False)
    results = {
        "prediction_type": prediction_type,
        "reg_error_pct": reg_error_pct,
        "reg_error_pct_int": int(reg_error_pct),
        "daily_growth_pct": daily_growth_pct,
    }
    return country_info, results



def regress_predict(prediction_type, country_info, data_name):
    applied_func = get_applied_func(prediction_type, country_info, data_name)
    updated_country_info, results = \
        add_linear_regression_log_and_prediction(
            country_info, data_name, applied_func, prediction_type)
    return updated_country_info, results

def get_latest_value(pd_series):
    return pd_series.dropna(how="any")[-1]

def regress_predict_data(data_name, country_info, is_peak):
    prediction_types = ["Logistics", "Exponential"]
    if is_peak: # forbid logistics+exp if peak is too close
        prediction_types.append("Logistics + Exponential")
        prediction_types.append("Logistics(Incr) + Logistics(Decr)")
        prediction_types.append("Logistics + Linear")
    # todo: add predictions to country_info as pointer
    models_results = []
    for prediction_type in prediction_types:
        try:
            updated_country_info, results = \
                regress_predict(prediction_type, country_info, data_name)
            country_info = updated_country_info
            models_results.append(results)
        except RuntimeError as error:
            # RuntimeError shall happen when curve_fit doesn't find any parameter
            print("    Couldn't find a fit for {} ({})".format(prediction_type, error))
            pass

    model_results_best = \
        sorted(models_results, key = lambda result: result["reg_error_pct"])[0]
    print("    Chose {} regression (error={:.1f})"
          .format(model_results_best["prediction_type"],
                  model_results_best["reg_error_pct"]))

    column_name_prediction = \
        get_column_name_func(data_name, model_results_best["prediction_type"], False, True)
    prediction = max(0, int(get_latest_value(country_info[column_name_prediction])))
    return country_info, {
        **model_results_best,
        "last_update": int(get_latest_value(country_info[data_name])),
        "is_peak_str": "Yes" if is_peak else "No",
        "prediction": prediction,
    }

# Plot ----------------------------------
def plot_country_log(country, all_results, country_info, log_scale):
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
    try:
        return improved_country_names[country_name]
    except KeyError as e:
        return country_name

def process_plot_country(country, country_info):
    country_all_results = {}

    for data_name in data_names:
        print("Analyzing {}".format(data_name))
        country_info[data_name + "Smooth"] = smooth_curve(country_info[data_name])
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

    image_name_log = plot_country_log(country, country_all_results, country_info, True)
    image_name_normal = plot_country_log(country, country_all_results, country_info, False)
    index_str_list = [str(timestamp) for timestamp in country_info.index.tolist()]

    def export_data(data_name):
        return country_info[data_name].values.tolist()
    def export_data_prediction(data_name):
        column_name = get_column_name_func(
            data_name, country_all_results[data_name]["prediction_type"], False, True)
        return export_data(column_name)

    country_all_results = {
        "country_data": country_all_results,
        "country": improve_country_name(country),
        "image_name_log": image_name_log,
        "image_name_normal": image_name_normal,
        "dates": index_str_list,
        "new_confirmed": export_data(get_cases_name()),
        "new_deaths": export_data(get_deaths_name()),
        "prediction_confirmed": export_data_prediction(get_cases_name()),
        "prediction_deaths": export_data_prediction(get_deaths_name()),
    }
    return country_all_results

def save_json(file_name, content):
    with open(file_name, "w") as outfile:
        json.dump(content, outfile)

def get_countries(world_info):
    if args.favorite:
        return favorite_countries
    elif args.all:
        not_favorite_countries = set(all_countries) - set(favorite_countries)
        return favorite_countries + list(not_favorite_countries)
    else:
        return [args.country]

images_info = []
countries = get_countries(world_info)
for index, country_name in enumerate(countries):
    try:
        country_info = get_country_info(country_name)
        if country_info is None:
            print("No case found for {}".format(country_name))
            continue
        print("Processing {} - {} cases ({}/{})"
              .format(country_name, countries_max_cases_dict[country_name], index + 1, len(countries)))
        image_info = process_plot_country(country_name, country_info)
        images_info.append(image_info)
    except (ValueError, IndexError) as error:
        print("No case found for {} (error: {})".format(country_name, error))
        continue
    print()

global_info = {
    "days_predict": args.days_predict,
    "favorite_countries": favorite_countries,
    "min_new_cases": min_new_cases,
    "min_total_cases": min_total_cases,
    "date_last_update": get_today_date_str(),
}

save_json(os.path.join("docs", "_data", "images_info.json"), images_info)
save_json(os.path.join("docs", "_data", "global_info.json"), global_info)

if args.show and images_info:
    plt.show()

if args.publish or args.publish_push:
    push_if_outdated(args.publish_push, former_date)
