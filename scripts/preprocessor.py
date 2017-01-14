# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:37:37 2017

@author: Joost
"""

import data_reader as dr
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class PreProcessor:
    
    def __init__(self, feature_type='cv', remove_stopwords=False):
        """
        @var feature_type: Determines how to convert the words to features.
        'cv': CountVector
        'tfidf': TF-IDF
        @var remove_stopwords: Can be set to true for removing stop words.
        """
        self.X, self.y = dr.load_data()
        self.feature_type = feature_type
        self.cv = CountVectorizer()
        self.tfidf = TfidfTransformer()
        
    def get_features(self):
        """
        Returns:
            X: Numpy 2D vector with a feature-representation of all tweets.
            y: Numpy vector with whether the tweet is a rumour (1) or not (0).
        This method uses either CountVector or tfidf to convert the tweets to features.
        CountVector: Words of tweet will be indicated with 1 at their 
            respective places, all other will be 0.
        Number of features = vocabulary size.
        X_cv_features = np.array(self.cv.fit_transform(self.X).todense())
        """
        X_cv_features = self.cv.fit_transform(self.X)
        if self.feature_type == 'cv':
            return np.array(X_cv_features.todense()), np.array(self.y)
        elif self.feature_type == 'tfidf':
            X_tfidf_features = self.tfidf.fit_transform(X_cv_features)
            return np.array(X_tfidf_features.todense()), np.array(self.y)
        else:
            raise Exception("Feature type is not 'cv' nor 'tfidf'.")
            
    def use_cv(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.feature_type = 'cv'
        
    def use_tfidf(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.feature_type = 'tfidf'