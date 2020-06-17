""" Math related utilities, including math functions for curve fitting 
"""


import numpy as np
from sklearn.metrics import mean_absolute_error


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

def mean_absolute_error_norm(X, Y):
    return mean_absolute_error(X, Y)/ np.mean(X) * 100

def get_float_index(country_info_ranged):
    return np.linspace(0, 1, len(country_info_ranged.index))

# models functions  ---------------------

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
