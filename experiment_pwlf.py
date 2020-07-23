from pprint import pprint
import numpy as np
from fetch_data_class import FakeArgs
from fetch_excel import ExcelFetcher
from fetch_apis import APIFetcher
import matplotlib.pyplot as plt
from math_utils import series_to_float, smooth_curve
import pwlf  # https://jekel.me/piecewise_linear_fit_py/

def is_post_peak_slopes(slopes):
    # print("slopes 0:{} 1:{}".format(slopes[0], slopes[1]))
    return slopes[0] > 0. and slopes[1] < 0. and slopes[1] < slopes[2]

def predict_pwlf(country_data_series, index_float, index_float_extended, number_of_breakpoints):
    """ country_data is a numpy array
    """
    country_data = series_to_float(country_data_series)
    # first_non_zero = (country_data!=0).argmax(axis=0)
    first_non_zero = (country_data != 0).argmax()
    # country_data_non_zero = country_data_series[first_non_zero:]
    country_data_non_zero = country_data_series
        # pwlf needs y[0] = 0. for some unknown reason, so cut the initial zeroes
    ones = 0. * series_to_float(country_data_non_zero) + 1.
    # y = smooth_curve(np.maximum(np.log(series_to_float(country_data_non_zero)), ones))
    y = np.log(np.maximum(smooth_curve(series_to_float(country_data_non_zero), 7), ones))
    y_0 = y[0]
    y_pwlf = y - ones * y_0 # pwlf needs y[0] = 0. for some unknown reason
    # index_float_non_zero = index_float[first_non_zero:]
    index_float_non_zero = index_float
    pwlf_fitter = pwlf.PiecewiseLinFit(index_float_non_zero, y_pwlf)
    breaks = pwlf_fitter.fit(number_of_breakpoints)
    slopes = pwlf_fitter.calc_slopes()

    def predict_on_index(index_as_float):
        x_hat = np.linspace(index_as_float.min(), index_as_float.max(), len(index_as_float))
        prediction_log = pwlf_fitter.predict(x_hat) + y_0  # adding back y[0] after setting y[0] = 0. previously
        # print(len(country_data_series[:first_non_zero] + np.exp(prediction_log)))
        # return np.concatenate([country_data_series[:first_non_zero], np.exp(prediction_log)])
        return np.exp(prediction_log)

    return predict_on_index(index_float_extended), \
        predict_on_index(index_float_non_zero), \
        is_post_peak_slopes(slopes)

def plot_pwlf(country_data, index_float, index_float_extended, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(index_float, country_data)
    # plt.plot(x_hat, prediction, '-')
    plt.plot(index_float_extended, prediction, '-')
    ax.set_yscale("log")
    plt.show()


def predict_plot_pwlf(country_data, index_float, index_float_extended):
    _, prediction = predict_pwlf(country_data, index_float, index_float_extended, 3)
    plot_pwlf(country_data, index_float, index_float_extended, prediction)

if __name__ == '__main__':

    fake_args = FakeArgs()
    # data_fetcher = ExcelFetcher(fake_args, True)
    # country_dataframe = data_fetcher.get_country_info("France")
    data_fetcher = APIFetcher(fake_args)
    country_name = "France"
    # country_name = "USA"
    # country_name = "Belgium"
    # country_name = "Germany"

    country_dataframe = data_fetcher.get_country_info(country_name)
    country_data = country_dataframe[data_fetcher.get_deaths_name()].dropna(how="any")
    index_float = series_to_float(country_data.index)

    predict_plot_pwlf(country_data, index_float, index_float)
    # date_after_peak = "2020-4-1"
    # country_data_cut = country_dataframe.loc[date_after_peak:]
    # predict_plot_pwlf(country_data_cut, 2)
