# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:39:36 2017

Imports the tweets that can contains that are classified as either rumour
or nonrumour, and replaces this with a 0 or 1.

@author: Joost
"""

import csv
from collections import OrderedDict

def remove_dups(first_array, second_array):
    
    already_visited = OrderedDict()

    for fa1, fa2 in list(zip(first_array, second_array)):
        if fa1 not in already_visited.keys():
            already_visited[fa1] = fa2
            
    return list(already_visited.keys()), list(already_visited.values())
            
def load_data():
    """
    Loads the data from tweets.csv and returns all tweets that contain either
    a 'NR' or an 'R', that is replaced by 0 and 1 respectively.
    """
    tweet_text = []
    tweet_rumour = []
    
    # Mine turtle!
    with open('../data/tweets.csv', newline='', encoding='utf8') as csvfile:
        tweets = csv.reader(csvfile, delimiter='\t')
        for tweet in tweets:
            tweet_text.append(tweet[3])
            tweet_rumour.append(tweet[-1])
    
    tweet_text, tweet_rumour = zip(*((x, y) for x, y in zip(tweet_text, tweet_rumour) if 'R' in y))
    tweet_rumour = [0 if tw_r == 'NR' else 1 for tw_r in tweet_rumour]
    
    tweet_text, tweet_rumour = remove_dups(tweet_text, tweet_rumour)
    
    return tweet_text, tweet_rumour