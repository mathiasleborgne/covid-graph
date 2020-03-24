import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--reload", help="reload xlsx", action="store_true")
parser.add_argument("--start_date", help="Date in format 2020-3-1", default='2020-3-1')
parser.add_argument("--country", help="the country", default='France')
args = parser.parse_args()

# https://www.data.gouv.fr/fr/datasets/cas-confirmes-dinfection-au-covid-19-par-region/
url_input = "https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide"
file_name_output = "COVID-19-geographic-disbtribution-worldwide.xlsx"
fetch_excel = True
folder_xlsx = "."

def get_html_text(url_input):
    # get html from url
    request = requests.get(url_input)
    return request.text

def get_xlsx_url(html_text):
    # get xlsx address by taking 1st .xlsx ocurrence
    for line in html_text.split("\n"):
    # example: <div class="media-left"><i class="media-object fa fa-download" aria-hidden="true"></i></div><div class="media-body"><a href="https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-03-21.xlsx" type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet; length=232484" title="Open file in new window" target="_blank" data-toggle="tooltip" data-placement="bottom">Download today’s data on the geographic distribution of COVID-19 cases worldwide</a><span> - EN - [XLSX-227.04 KB]</span></div>
        if "media-object" in line and ".xlsx" in line:
            return line.split("href=\"")[1].split("\" type=")[0]
    return None

def save_xlsx(xlsx_url, file_name_output):
    # download xlsx from .xlsx url
    # https://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
    response = requests.get(xlsx_url, stream=True)
    path_output = os.path.join(folder_xlsx, file_name_output)
    with open(path_output, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

def fetch_excel():
    html_text = get_html_text(url_input)
    xlsx_url = get_xlsx_url(html_text)
    save_xlsx(xlsx_url, file_name_output)


if args.reload:
    fetch_excel()
try:
    world_info = pd.read_excel(file_name_output)
except FileNotFoundError as e:
    fetch_excel()
    world_info = pd.read_excel(file_name_output)

world_info['date'] = world_info['DateRep']
world_info = world_info.set_index(['date'])
world_info.sort_values(by='date')
print("Countries:", sorted(set(world_info['Countries and territories'])))
country = args.country
country_info = world_info[world_info['Countries and territories'].isin([country])]
country_info = country_info.loc[:args.start_date]

# log10
def log10_filter(x):
    if x <= 1:
        return 0.
    else:
        return np.log10(x)

country_info_log = country_info.copy()
country_info_log["Cases"] = country_info_log["Cases"].apply(log10_filter)
country_info_log["Deaths"] = country_info_log["Deaths"].apply(log10_filter)
print(country_info_log)

# linear regression
# X = pd.to_datetime(country_info_log.iloc[:, 0]).astype(int).values.reshape(-1, 1)  # values converts it into a numpy array
# # X = range(len(X_origin)).reshape(-1, 1)
# print(country_info_log.index)
# Y = country_info_log.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
# print(X)
# assert len(X) == len(X_origin)

# print(Y)
# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions

# Plot
ax = country_info_log.plot(x='DateRep', y=['Cases', 'Deaths'])
plt.xlabel("date")
plt.ylabel("log_10")

folder_images = "saved_images"
plt.savefig(os.path.join(folder_images, 'img_log10_{}.png'.format(country)))
plt.show()