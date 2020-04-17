import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import json
import math
import os
import datetime
import numpy as np
import scipy
import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse

from fetch_excel import fetch_excel


""" This script:
        - gets an Excel file with all countries information
        - makes a pandas dataframe, filtered by date
        - computes a curve fit on the figures with
            - exponential model
            - logistics model
            - logistics then exponential model (piecewise)
        - chooses best model and predicts a few days ahead
        - plots the figures and prediction

    Todo:
        - nav bar
        - more models
        - anchor links
        - index is returned at reindexing...why?
        - for logistics
            - fix daily growth
            - use past data
        - use https://covid19api.com/#details
"""
# constants --------------------------------------
min_cases = 100
min_days_post_peak = 8
min_decrease_post_peak = 10.
number_days_future_default = 7
favorite_countries = [
    "France",
    "Spain",
    "United_States_of_America",
    "United_Kingdom",
    "Italy",
    "Belgium",
    "Germany",
]
data_names = ["cases", "deaths"]

# parser -------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--reload", help="reload xlsx", action="store_true")
parser.add_argument("--start_date", help="Date in format 2020-3-1", default="2020-3-1")
parser.add_argument("--country", help="Select a specific country", default="France")
parser.add_argument("--favorite", help="Favorite countries", action="store_true")
parser.add_argument("--all", help="All countries", action="store_true")
parser.add_argument("--show", help="Show images", action="store_true")
parser.add_argument("--temp_curves", help="Show temporary curves", action="store_true")

parser.add_argument("--days_predict", help="Number of days to predict in the future", default=number_days_future_default, type=int)
args = parser.parse_args()


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

world_info["date"] = world_info["dateRep"]
world_info = world_info.set_index(["date"])
world_info.sort_values(by="date")
all_countries_world = set(world_info.countriesAndTerritories)


def add_future_index(country_info, number_days_future):
    dates_extended = pd.date_range(country_info.index[0], periods=number_days_future)
    ix_dates_extended = pd.DatetimeIndex(country_info.index).union(pd.DatetimeIndex(dates_extended))
    return country_info.reindex(ix_dates_extended)

def get_country_info(country):
    country_info = world_info[world_info["countriesAndTerritories"].isin([country])]
    return add_future_index(country_info.loc[:args.start_date], args.days_predict)

countries_max_cases_dict = {
    country: get_country_info(country)["cases"].max()
    for country in all_countries_world
}
# countries_population_dict = dict(zip(world_info.countriesAndTerritories, world_info.popData2018))
all_countries = [country
                 for country, max_cases in countries_max_cases_dict.items()
                 if max_cases > min_cases]
print("Countries:", sorted(all_countries))

# math utils -----------------------

def shift(xs, n):
    if n >= 0:
        return np.r_[np.full(n, np.nan), xs[:-n]]
    else:
        return np.r_[xs[-n:], np.full(-n, np.nan)]

def smooth_curve(y):
    n_pts_box = 6
    box = np.ones(n_pts_box)/n_pts_box

    y_smooth = np.convolve(y, box, mode='same')
    # y_smooth_shifted = shift(y_smooth, -int(n_pts_box/2))
    # no need to shift
    return y_smooth

def smooth_max(country_info, data_name):
    y_smooth = country_info[data_name + "Smooth"]
    max_y = y_smooth.max()
    argmax_y = y_smooth.idxmax(axis=1)
    return max_y, argmax_y

def logistics_full(x, a, b, l_max):
    return l_max/(1+np.exp(a*x+b))

def logistics_exp_full(x, a_log, b_log, a_exp, b_exp, l_max, argmax_float):
    """ exponential is constrained on b_exp, considering the exponential starts
        at l_max:
        In log: ln(y) - ln(l_max) = a_exp*(x - argmax_peak)
        In not-log: y = exp(a_exp*(x - argmax_peak) + ln(l_max))
    """
    return np.piecewise(x, [x <= argmax_float , x > argmax_float],
                        [lambda x: logistics_full(x, a_log, b_log, l_max),
                         lambda x: exponential_full(x - argmax_float, a_exp, np.log(l_max))])

def exponential_full(x, a, b):
    return np.exp(a*x+b)

def get_float_index(country_info_ranged):
    return np.linspace(0, 1, len(country_info_ranged.index))


def get_peak_date(country_info, data_name):
    max_country, argmax_country = smooth_max(country_info, data_name)
    country_data = country_info[data_name].dropna(how="any")
    country_data_smooth = country_info[data_name + "Smooth"].dropna(how="any")

    today = country_data.index[-1]
    days_after_peak = (today - argmax_country).days
    latest_value = country_data_smooth[-1]
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
    country_info_ranged = country_info
    country_info_filtered = country_info.dropna(how="any")
    X = country_info_filtered.index.to_numpy(dtype=np.float32).reshape(-1, 1).ravel()
    X_extended = get_float_index(country_info_ranged)
        # todo back to timestamps?
    X = X_extended[:len(country_info_filtered.index)]
    Y = country_info_filtered[data_name].to_numpy().reshape(-1, 1).ravel()

    popt, pcov = scipy.optimize.curve_fit(applied_func, X, Y)
    applied_func_params = lambda x: applied_func(x, *popt)

    reg_error_pct = mean_absolute_error(Y, applied_func_params(X))/ np.mean(Y) * 100

    # predict
    # X_extended = country_info_ranged.index.to_numpy(dtype=np.float32).reshape(-1, 1).ravel()
    Y_pred = applied_func_params(X_extended)

    # compute daily multiplicative factor
    # eg for Exponential:
    # ln(Y) = a*t+b --> Y(t) = B*exp(a*t) --> Y(t+1) = B*exp(a)*exp(a*t) = exp(a)*Y(t)
    daily_growth_pct = (applied_func_params(X[-1]) - applied_func_params(X[-2]))/applied_func_params(X[-1]) * 100
    # print("{} grow of {} pct each day".format(data_name, daily_growth_pct))

    # add to dataframe
    prediction_log = pd.Series(Y_pred.ravel(),
                               name=get_column_name_func(data_name, prediction_type, True, True),
                               index=country_info_ranged.index)
    prediction = pd.Series(Y_pred.ravel(),
                           name=get_column_name_func(data_name, prediction_type, False, True),
                           index=country_info_ranged.index)
    country_info = pd.concat([country_info, prediction, prediction_log],
                              axis=1, sort=False)
    results = {
        "prediction_type": prediction_type,
        "reg_error_pct": reg_error_pct,
        "reg_error_pct_int": int(reg_error_pct),
        "daily_growth_pct": daily_growth_pct,
    }
    return country_info, results


def get_applied_func(prediction_type, country_info, data_name):
    """ Get the model function for curve fitting.
        Some models need some additional information to work better (maximum, argmax)
    """
    l_max, argmax_country = smooth_max(country_info, data_name)
    logistics_maxed = lambda x, a, b: logistics_full(x, a, b, l_max)
    if prediction_type == "Logistics":
        return logistics_maxed
    elif prediction_type == "Logistics+Exponential":
        index_float = get_float_index(country_info)
        argmax_loc = country_info.index.get_loc(argmax_country)
        argmax_float = index_float[argmax_loc]
        return lambda x, a_log, b_log, a_exp, b_exp: \
            logistics_exp_full(x, a_log, b_log, a_exp, b_exp, l_max, argmax_float)
    else:
        return exponential_full

def regress_predict(prediction_type, country_info, data_name):
    applied_func = get_applied_func(prediction_type, country_info, data_name)
    updated_country_info, results = \
        add_linear_regression_log_and_prediction(
            country_info, data_name, applied_func, prediction_type)
    return updated_country_info, results

def get_latest_value(pd_series):
    return pd_series.dropna(how="any")[-1]
    """ Get the model function for curve fitting.
        Some models need some help to work better (maximum, argmax)
    """

def regress_predict_data(data_name, country_info, is_peak):
    prediction_types = ["Logistics", "Exponential"]
    if is_peak: # forbid logistics+exp if peak is too close
        prediction_types.append("Logistics+Exponential")
    # todo: add predictions to country_info as pointer
    models_results = []
    for prediction_type in prediction_types:
        updated_country_info, results = \
            regress_predict(prediction_type, country_info, data_name)
        country_info = updated_country_info
        models_results.append(results)

    model_results_best = \
        sorted(models_results, key = lambda result: result["reg_error_pct"])[0]
    print("    Chose {} regression (error={:.1f})"
          .format(model_results_best["prediction_type"],
                  model_results_best["reg_error_pct"]))

    column_name_prediction = \
        get_column_name_func(data_name, model_results_best["prediction_type"], False, True)
    prediction = int(get_latest_value(country_info[column_name_prediction]))
    return country_info, {
        **model_results_best,
        "last_update": int(get_latest_value(country_info[data_name])),
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
        .plot(x="index", y=["cases", "deaths"] + prediction_columns_names)
    # ax = country_info.reset_index().plot(x="index", y=["casesLog", "deathsLog", "PredictionLog"])
    # ax = country_info.reset_index().plot(x="index", y=["cases", "deaths"])
    if log_scale:
        ax.set_yscale("log")
    plt.xlabel("date")

    # plt.ylabel("log_10")
    if args.show:
        plt.title("{} - Cases/Deaths\n(Reg. error: {:.1f} pct / Daily growth: {:.1f} pct)"
                  .format(country, all_results["cases"]["reg_error_pct"],
                          all_results["cases"]["daily_growth_pct"]))
    folder_images = "saved_images"
    image_name = "img_log10_{}_{}.png".format(country, "log" if log_scale else "normal")
    plt.savefig(os.path.join(folder_images, image_name))
    plt.savefig(os.path.join("docs", "assets", "img", image_name))
    return image_name

def process_plot_country(country):
    country_info = get_country_info(country)
    country_all_results = {}

    for data_name in data_names:
        print("Analyzing {}".format(data_name))
        country_info[data_name + "Smooth"] = smooth_curve(country_info[data_name])
        peak_date = get_peak_date(country_info, data_name) # needs smooth column addition...
        is_peak = peak_date is not None
        print("    is peak? {} - date max: {}".format(is_peak, peak_date))
        start_date = pd.Timestamp(args.start_date)
        prediction_date = pd.Timestamp(country_info.index[-1])
            # todo: not same type...
        updated_country_info, country_results_data = \
            regress_predict_data(data_name, country_info, is_peak)
        country_all_results[data_name] = country_results_data
        country_info = updated_country_info

    image_name_log = plot_country_log(country, country_all_results, country_info, True)
    image_name_normal = plot_country_log(country, country_all_results, country_info, False)
    country_all_results = { **country_all_results,
        "country": country,
        "image_name_log": image_name_log,
        "image_name_normal": image_name_normal,
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
for index, country in enumerate(countries):
    try:
        print("Processing {} ({}/{})".format(country, index + 1, len(countries)))
        image_info = process_plot_country(country)
        images_info.append(image_info)
    except ValueError as error:
        print("No case found for {} (error: {})".format(country, error))
        continue
    print()

global_info = {
    "days_predict": args.days_predict,
    "favorite_countries": favorite_countries,
    "min_cases": min_cases,
    "date_last_update": datetime.date.today().strftime("%B %d, %Y"),
}

save_json(os.path.join("docs", "_data", "images_info.json"), images_info)
save_json(os.path.join("docs", "_data", "global_info.json"), global_info)

if args.show and images_info:
    plt.show()
