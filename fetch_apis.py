import requests
import pandas as pd
from pprint import pprint


def get_all_countries_info_by_api():
    all_countries_command = "https://corona-api.com/countries"
    response = requests.get(all_countries_command)
    obj_response = response.json()
    return obj_response["data"]


def get_country_by_api(country_code):
    country_command = "http://corona-api.com/countries/{}".format(country_code)
    response = requests.get(country_command)
    obj_response = response.json()
    country_data = obj_response["data"]
    country_timeline = country_data["timeline"]
    if not country_timeline:  # empty list
        print("{}: no country timeline (data: {})".format(country_data["name"], country_data))
        return None
    # don't take today's data
    country_timeline = country_timeline[1:]
    country_df = pd.DataFrame.from_dict(country_timeline)
    country_df["date"] = country_df["date"].apply(pd.to_datetime)
    country_df = country_df.set_index(["date"])
    return country_df

if __name__ == '__main__':
    pprint(get_country_by_api("FR"))
