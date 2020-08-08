""" Utilities to process, fit curves and make predictions on time series 
"""

import scipy
import pandas as pd
import numpy as np
from math_utils import get_applied_func, get_float_index, \
    mean_absolute_log_error_norm, series_to_float, quick_prediction_plot, \
    predict_pwlf

def get_column_name_func(column_name, prediction_type, is_inverted, is_prediction):
    """ get the name of the column in pandas frame for data, prediction, data with function applied (inverted)
    """
    # todo: inverted and prediction: compatible?
    column_name_func = column_name + prediction_type
    if is_prediction:
        column_name_func = column_name_func + "Prediction"
    if is_inverted:
        column_name_func = column_name_func + "Inv"
    return column_name_func

# regression -----------------------------
def make_prediction_error_growth(X, X_extended, Y, prediction_type, applied_func, country_data_filtered):
    """ make curve fitting, extend it to make prediction 
        Compute regression error and daily growth 
    """
    error_penalty = 0.

    if prediction_type == "LogPiecewiseLinearFit3":
        Y_pred, func_applied_on_index, is_post_peak_slopes = predict_pwlf(country_data_filtered, X, X_extended, 3)
        # todo: predict_pwlf shouldn't use country_data_filtered
        daily_growth_pct= 0. # todo
        # print("is_post_peak_slopes", is_post_peak_slopes)
        if not is_post_peak_slopes:
            # bigger error if slopes don't correspond to growth + decrease + growth/decrease
            error_penalty = 20.
    else:
        popt, pcov = scipy.optimize.curve_fit(applied_func, X, Y)
        applied_func_params = lambda x: applied_func(x, *popt)
        func_applied_on_index = applied_func_params(X)

        # predict
        # X_extended = country_data.index.to_numpy(dtype=np.float32).reshape(-1, 1).ravel()
        Y_pred = applied_func_params(X_extended)

        # compute daily multiplicative factor
        # eg for Exponential:
        # ln(Y) = a*t+b --> Y(t) = B*exp(a*t) --> Y(t+1) = B*exp(a)*exp(a*t) = exp(a)*Y(t)
        daily_growth_pct = (applied_func_params(X[-1]) - applied_func_params(X[-2]))/applied_func_params(X[-1]) * 100
    # print("{} grow of {} pct each day".format(data_name, daily_growth_pct))
    # quick_prediction_plot(country_data_filtered, X, X_extended, Y_pred)
    reg_error_pct = mean_absolute_log_error_norm(Y, func_applied_on_index) + error_penalty
    return Y_pred, reg_error_pct, daily_growth_pct

def add_linear_regression_log_and_prediction(
    country_info, data_name, applied_func, prediction_type):
    """ preprocess data for prediction (extend range, index as floats) 
        Pack the results in a JSON file 
    """
    column_applied_func = get_column_name_func(data_name, prediction_type, True, False)
    # start_date, end_date = date_range
    # country_info_ranged = country_info.loc[start_date:end_date]
    # country_info_filtered = country_info.loc[start_date:end_date].dropna(how="any")
    country_data = country_info[data_name]
    country_data_filtered = country_data.dropna(how="any")
    X = series_to_float(country_data_filtered.index)
    X_extended = get_float_index(country_data)
        # todo back to timestamps?
    X = X_extended[:len(country_data_filtered.index)]
    Y = series_to_float(country_data_filtered)
    Y_pred, reg_error_pct, daily_growth_pct = \
        make_prediction_error_growth(X, X_extended, Y, prediction_type,
                                     applied_func, country_data_filtered)
    # add to dataframe
    prediction_series = pd.Series(np.round(Y_pred).ravel(),
                           name=get_column_name_func(data_name, prediction_type, False, True),
                           index=country_data.index)
    country_info = pd.concat([country_info, prediction_series],
                              axis=1, sort=False)
    results = {
        "prediction_type": prediction_type,
        "reg_error_pct": reg_error_pct,
        "reg_error_pct_int": int(reg_error_pct),
        "daily_growth_pct": daily_growth_pct,
    }
    return country_info, results


def regress_predict(prediction_type, country_info, data_name):
    """ Fit curve, predict and give results for a type of prediction (model)
        Add results to data frame and give results
    """
    applied_func = get_applied_func(prediction_type, country_info, data_name)
    updated_country_info, results = \
        add_linear_regression_log_and_prediction(
            country_info, data_name, applied_func, prediction_type)
    return updated_country_info, results

def get_latest_value(pd_series):
    """ Get the latest values of a pandas series, Nan excluded
    """
    return pd_series.dropna(how="any")[-1]

def regress_predict_data(data_name, country_info, is_peak):
    """ get country info with predictions added, plus some country-level info for JSON export
    """
    prediction_types = ["Logistics", "Exponential", "LogPiecewiseLinearFit3"]
    # LogPiecewiseLinearFit3 is OK without peak because of the USA example with 1st low peak, then higher rebound
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
