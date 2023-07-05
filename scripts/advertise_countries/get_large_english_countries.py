"""
Author: Joshua Ashkinaze
Date: 2023-07-03

Description: This script gets the list of the 20 countries with largest English speaking pop, where over half
of the pop speaks English. It uses BeautifulSoup to scrape the data from Wikipedia from:
https://en.wikipedia.org/wiki/List_of_countries_by_English-speaking_population

Note: Numbers may change next year.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import logging


def convert_to_integers(number):
    number = number.replace(",", "")
    number = number.replace("+", "")
    return int(number)


def main():
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    data = []
    url = "https://en.wikipedia.org/wiki/List_of_countries_by_English-speaking_population"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="wikitable sortable")
    headers = table.find_all("th")
    header_data = [header.get_text(strip=True) for header in headers]
    for row in table.find_all("tr")[2:]:
        cells = row.find_all("td")
        row_data = [cell.get_text(strip=True) for cell in cells]
        data.append(row_data)

    df = pd.DataFrame(data)
    df.columns = ["country", "n", "english_n", "english_prop", "firstlang_n", "firstlang_prop", "adlang_n",
                  "adlang_prop", "notes"]
    df = df.query("country != 'World'")
    df['english_n'] = df['english_n'].apply(convert_to_integers)
    df['n'] = df['n'].apply(convert_to_integers)
    df['english_prop'] = df['english_n'] / df['n']
    large_english = df.query("english_prop > 0.75").sort_values(by=['english_n'], ascending=False).head(15)
    logging.info(large_english[['english_n', 'english_prop', 'country']].to_string())
    logging.info("\n" + str(sorted(large_english['country'].tolist())))


if __name__ == "__main__":
    main()
