import argparse
import json
from constants import number_days_future_default

# parser -------------------------------------------
def get_args():
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
    return parser.parse_args()

def save_json(file_name, content):
    """ Save as JSON file
    """
    with open(file_name, "w") as outfile:
        json.dump(content, outfile, indent=4)

