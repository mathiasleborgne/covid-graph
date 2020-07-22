from pprint import pprint
from fetch_data_class import FakeArgs
from fetch_excel import ExcelFetcher
from fetch_apis import APIFetcher
import matplotlib.pyplot as plt
from math_utils import series_to_float, predict_pwlf


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
