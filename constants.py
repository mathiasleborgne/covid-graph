""" Constants for plot_covid script, including parser default values, settings, etc.
"""

import os

min_new_cases = 100 # minimum number of cases to start displaying a curve
min_total_cases = 1000 # minimum number of Total cases to display a country
min_cases_start_date = 30  # minimum number of cases to start the curve display
min_days_post_peak = 12 # minimum number of decrease days after smooth curve peak
    # in order to actually detect "the" peak
min_decrease_post_peak = 10.  # minimum decrease un cases/deaths to detect a peak, in percents
number_days_future_default = 15 # number of days ahead for prediction
max_countries_display = 50  # max number of countries to display
default_smoothing_length = 7 # default length of smoothing window, 1 week as week-ends are often counted differently

improved_country_names = {
    # when you get a bad name like those on the left,
    # replace by bame on the right
    "USA": "United States of America",
    "UK": "United kingdom",
    "S. Korea": "South Korea",
}

path_country_data_json = os.path.join("docs", "_data", "images_info.json")  # path for Jekyll's JSON files, where data is exported by the script
