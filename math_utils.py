""" Math related utilities, including math functions for curve fitting
"""
from constants import default_smoothing_length
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pwlf  # https://jekel.me/piecewise_linear_fit_py/


# math utils -----------------------

def shift(xs, n):
    """ Shift a numpy array of n steps (n>=0 or n<0)
    """
    if n >= 0:
        return np.r_[np.full(n, np.nan), xs[:-n]]
    else:
        return np.r_[xs[-n:], np.full(-n, np.nan)]

def smooth_curve(y, n_pts_box=default_smoothing_length):
    # n_pts_box: width of the window
    box = np.ones(n_pts_box)/n_pts_box
    y_smooth = np.convolve(y, box, mode='same')
    # y_smooth_shifted = shift(y_smooth, -int(n_pts_box/2))
    # no need to shift
    return y_smooth

def smooth_max(country_info, data_name):
    """ return max and argmax for smoothened curve 
        It needs to be already smoothened 
    """ 
    y_smooth = country_info[data_name + "Smooth"]
    max_y = y_smooth.max()
    argmax_y = y_smooth.idxmax(axis=1)
    return max_y, argmax_y

def mean_absolute_error_norm(X, Y):
    return mean_absolute_error(X, Y)/ np.mean(X) * 100

def log_no_nan(X):
    return np.log(np.maximum(X, 0.*X +1)) #todo: ones?

def mean_absolute_log_error_norm(X, Y):
    return mean_absolute_error_norm(log_no_nan(X), log_no_nan(Y))

def get_float_index(country_info_ranged):
    """ Transforms a date index into a np array in [0;1] to be usable by regression algorithms
    """
    return np.linspace(0, 1, len(country_info_ranged.index))

# models functions  ---------------------

def logistics_full(x, a, b, l_max):
    """ Logistics function with all parameters. Can process np arrays. 
    """
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

def logistics_incr_decr_full(x, a_log1, b_log1, a_log2, b_log2, l_max, argmax_float):
    """ Logistics increasing before peak, decreasing after
    """
    return np.piecewise(x, [x <= argmax_float , x > argmax_float],
                        [lambda x: logistics_full(x, a_log1, b_log1, l_max),
                         lambda x: logistics_full(-x, a_log2, b_log2, l_max)])

def logistics_linear_full(x, a_log1, b_log1, a_lin, l_max, argmax_float):
    """ Logistics increasing before peak, linear decrease after
    """
    return np.piecewise(x, [x <= argmax_float , x > argmax_float],
                        [lambda x: logistics_full(x, a_log1, b_log1, l_max),
                         lambda x: a_lin * (x - argmax_float) + l_max])


def exponential_full(x, a, b):
    return np.exp(a*x+b)


def get_applied_func(prediction_type, country_info, data_name):
    """ Get the model function for curve fitting.
        Some models need some additional information to work better (maximum, argmax)
    """
    l_max, argmax_country = smooth_max(country_info, data_name)
    logistics_maxed = lambda x, a, b: logistics_full(x, a, b, l_max)
    if prediction_type == "Logistics":
        return logistics_maxed
    elif prediction_type == "Logistics + Exponential":
        index_float = get_float_index(country_info)
        argmax_loc = country_info.index.get_loc(argmax_country)
        argmax_float = index_float[argmax_loc]
        return lambda x, a_log, b_log, a_exp, b_exp: \
            logistics_exp_full(x, a_log, b_log, a_exp, b_exp, l_max, argmax_float)
    elif prediction_type == "Logistics(Incr) + Logistics(Decr)":
        index_float = get_float_index(country_info)
        argmax_loc = country_info.index.get_loc(argmax_country)
        argmax_float = index_float[argmax_loc]
        return lambda x, a_log1, b_log1, a_log2, b_log2: \
            logistics_incr_decr_full(x, a_log1, b_log1, a_log2, b_log2, l_max, argmax_float)
    elif prediction_type == "Logistics + Linear":
        index_float = get_float_index(country_info)
        argmax_loc = country_info.index.get_loc(argmax_country)
        argmax_float = index_float[argmax_loc]
        return lambda x, a_log1, b_log1, a_lin: \
            logistics_linear_full(x, a_log1, b_log1, a_lin, l_max, argmax_float)
    else:
        return exponential_full

def series_to_float(data_series):
    return data_series.to_numpy(dtype=np.float32).reshape(-1, 1).ravel()


def quick_prediction_plot(country_data, index_float, index_float_extended, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(index_float, country_data)
    # plt.plot(x_hat, prediction, '-')
    plt.plot(index_float_extended, prediction, '-')
    ax.set_yscale("log")
    plt.show()

# piecewise linear fit in log space ------------------------------------

def is_post_peak_slopes(slopes):
    # print("slopes 0:{} 1:{}".format(slopes[0], slopes[1]))
    if len(slopes) == 3 or len(slopes) == 4:
        is_rebound = slopes[0] > 0. and slopes[1] < 0. and slopes[1] < slopes[2]
        return is_rebound
    else:
        print("Warning: Bad slopes")
        return False

def predict_pwlf(country_data_series, index_float, index_float_extended, number_of_breakpoints):
    """ country_data_series is a pandas series
    """
    # todo: cut the initial array?
    country_data = series_to_float(country_data_series)
    first_non_zero = (country_data != 0).argmax()
    country_data_non_zero = country_data_series
    ones = 0. * series_to_float(country_data_non_zero) + 1.
    y = np.log(np.maximum(smooth_curve(series_to_float(country_data_non_zero), 7), ones))
    y_0 = y[0]
    y_pwlf = y - ones * y_0 # pwlf needs y[0] = 0. for some unknown reason
    index_float_non_zero = index_float
    pwlf_fitter = pwlf.PiecewiseLinFit(index_float_non_zero, y_pwlf)
    breaks = pwlf_fitter.fit(number_of_breakpoints)
    slopes = pwlf_fitter.calc_slopes()

    def predict_on_index(index_as_float):
        """ Make piecewise linear fit prediction on log space, then translate back to linear space
        """
        x_hat = np.linspace(index_as_float.min(), index_as_float.max(), len(index_as_float))
        prediction_log = pwlf_fitter.predict(x_hat) + y_0  # adding back y[0] after setting y[0] = 0. previously
        # print(len(country_data_series[:first_non_zero] + np.exp(prediction_log)))
        # return np.concatenate([country_data_series[:first_non_zero], np.exp(prediction_log)])
        return np.exp(prediction_log)

    return predict_on_index(index_float_extended), \
        predict_on_index(index_float_non_zero), \
        is_post_peak_slopes(slopes)


def get_error_with_smooth(country_info, data_name):
    """ mean absolute error between raw and smoothened curves
    """
    country_data_smooth = country_info[data_name + "Smooth"].dropna(how="any")
    len_smooth = len(country_data_smooth)
    country_data = country_info[data_name].dropna(how="any")[:len_smooth]
    return mean_absolute_error_norm(country_data_smooth, country_data)
