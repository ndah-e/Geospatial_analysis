#!/usr/bin/env python
# coding: utf-8

"""
    script to download cadastral files from the website https://finances.belgium.be/fr/experts_partenaires/plan-cadastral/lambert-72/2018
    require as input the url and the output folder to save the downloaded and unziped files
        first input file is the url 
        second input file the output directory
        
"""

import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import requests
from time import sleep
from random import randint

import pandas as pd
#import sys

import zipfile


def cadastral_plan_links(url):
    
    #url = "https://finances.belgium.be/fr/experts_partenaires/plan-cadastral/lambert-72/2018"
    response = requests.get(url, timeout=5)
    soup = BeautifulSoup(response.content, "html.parser")

    commune_name = []
    commune_url = []
    communes = soup.find('div', class_ = 'field-name-body field-type-text-with-summary').find_all("li")
    for com in communes:
        commune_name.append(com.a.get_text())
        commune_url.append(com.a.get('href'))
        
    cadastral_df = pd.DataFrame({'Commune':commune_name, 'url': commune_url})
    
    return cadastral_df


if __name__ == "__main__":
    
    #url = sys.argv[1]              # the URL from cadastral website that list the cadastral files for each commune
    #output_folder = sys.argv[2]    # output folder to store the downloaded files

    output_folder = "./data/cadastral/"
    output_folder_others = "./data/"
    url = "https://finances.belgium.be/fr/experts_partenaires/plan-cadastral/lambert-72/2018"
    
    cadastral_df = cadastral_plan_links(url)
    exclude = ['Toute la Belgique', 'Limites administratives']

    for i, row in cadastral_df.iterrows():
        sleep(randint(5, 40))
        if row.Commune in exclude:
            filename = output_folder_others + row.Commune + ".zip"
            filename = filename.replace(' ', '_')
        else:
            filename = output_folder + row.Commune + ".zip"
            
        url =  row.url.replace(' ', '%20')
        try:
            urllib.request.urlretrieve(url, filename)
            extract_folder = output_folder + row.Commune
            myzipfile = zipfile.ZipFile(filename)
            myzipfile.extractall(extract_folder)

        except urllib.HTTPError as e:
            if e.getcode() == 404: # check the return code
                print(row.Commune)
                continue
            raise # if other than 404, raise the error

