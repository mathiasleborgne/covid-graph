# covid-graph

This repo makes graphs for coronavirus cases from [this data source](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide)
to publish them to [this github page](https://mathiasleborgne.github.io/covid-graph/).

![Cases/Deaths 03/10/2020](ScreenshotApril10.png "Cases/Deaths 03/10/2020")


## Install and build

    # make website data
    pip install requirements.txt
    python plot_covid.py --all --reload
    # build website
    cd docs/
    bundle exec jekyll serve
    # open in browser http://127.0.0.1:4000/


## Commands

    usage: plot_covid.py [-h] [--reload] [--start_date START_DATE]
                         [--country COUNTRY] [--favorite] [--all] [--show]
                         [--days_predict DAYS_PREDICT]

    optional arguments:
      -h, --help            show this help message and exit
      --reload              reload xlsx
      --start_date START_DATE
                            Date in format 2020-3-1
      --country COUNTRY     Select a specific country
      --favorite            Favorite countries
      --all                 All countries
      --show                Show images
      --days_predict DAYS_PREDICT
                            Number of days to predict in the future
