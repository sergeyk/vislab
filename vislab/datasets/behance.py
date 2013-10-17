"""
Code to call the Behance API to construct a dataset.
"""

#import os
import requests
#import bs4
import pandas as pd
#import numpy as np
import random

import vislab

tags = ['photo','blue','car','chinese ink','colored pencil','colors','comic',
        'fashion illustration','graphics','infographic','landscape','vector',
        'watercolor']
testURL = 'http://www.behance.net//gallery/Icons/1140561'
projectNum = 1140561

def get_basic_dataset(force=False):
    """
    Return DataFrame of image_id -> page_url, artist_slug, artwork_slug.
    """
    filename = vislab.config['paths']['shared_data'] + \
        '/wikipaintings_basic_info.h5'
    df = vislab.util.load_or_generate_df(filename, fetch_basic_dataset, force)
    return df

def _getSmallest(sizeList):
    knownSizes = ['max_1240','max_1920','original']

    for s in knownSizes:
        if sizeList.has_key(s):
            return sizeList[s]

    print(sizeList)
    raise Exception

def fetch_single_project_image_URLs(projectNum):
    query = 'http://www.behance.net/v2/projects/'+str(projectNum)+'?api_key='+vislab.config['behanceAPIkey']
    print('fetching project %d, query %s'%(projectNum,query))

    r = requests.get(query)
    projectInfo = r.json()['project']
    imageData = filter(lambda x:x['type'] == 'image', projectInfo['modules'])
    return map(lambda x:_getSmallest(x['sizes']), imageData)

# set maximums to -1 in order to not have a maximum
def fetch_basic_dataset(maxRequests = 10, maxImagesPerProject=2):
    """
    Fetch basic info and page urls from a collection of projects.  
    Results are returned as a DataFrame.
    """
    print("Fetching Behance dataset.")

    projectList = pd.DataFrame.from_csv('behanceProjects.csv',header=-1)
    APIkey = vislab.config['behanceAPIkey']

    numRequests = 0

    random.seed(0)  # fix the seed so we get the same results each time

    imageData = []

    for index,row in projectList.iterrows():
        projectNum = row.name
        URL = row[1]
        label = row[2]

        imageURLs = fetch_single_project_image_URLs(projectNum)
        if len(imageURLs) <= maxImagesPerProject or maxImagesPerProject <= 0:
            pickedImageURLs = imageURLs
        else:
            pickedImageURLs = random.sample(imageURLs,maxImagesPerProject)
        
        for u in pickedImageURLs:
            imageData.append({'projectNum':projectNum,'projectURL':URL,'label':label,'imageURL':u})

        numRequests = numRequests + 1
        if maxRequests > 0 and numRequests==maxRequests:
            break

    df = pd.DataFrame(imageData)
    return df


if __name__ == '__main__':
    """
    Run the scraping with a number of workers taking jobs from a queue.
    """
    df = fetch_basic_dataset(maxRequests = 2, maxImagesPerProject=2)
    print(df)
