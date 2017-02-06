# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:37:37 2017

@author: Joost
"""

import data_reader as dr
import numpy as np
import sklearn.model_selection as ms
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
        if feature_type == 'bcv':
            self.use_binary = True
        else:
            self.use_binary = False
        self.feature_type = feature_type
        self.remove_stopwords = 'english' if remove_stopwords else None
        self.cv = CountVectorizer(stop_words=self.remove_stopwords, binary=self.use_binary)
        self.tfidf = TfidfTransformer()
        
    def get_features_and_labels(self):
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
        if self.feature_type == 'cv' or self.feature_type == 'bcv':
            return np.array(X_cv_features.todense()), np.array(self.y)
        elif self.feature_type == 'tfidf':
            X_tfidf_features = self.tfidf.fit_transform(X_cv_features)
            return np.array(X_tfidf_features.todense()), np.array(self.y)
        else:
            raise Exception("Feature type is not 'cv' nor 'tfidf'.")
            
    def get_all_sets(self, cv=True, random_seed=False):
        X_all, y_all = self.get_features_and_labels()
        if cv:
            if random_seed:
                X_train, X_test, y_train, y_test = ms.train_test_split(X_all, y_all, test_size=0.2, random_state=random_seed)
                X_train, X_cv, y_train, y_cv = ms.train_test_split(X_train, y_train, test_size=0.25, random_state=random_seed)
            else:
                X_train, X_test, y_train, y_test = ms.train_test_split(X_all, y_all, test_size=0.2)
                X_train, X_cv, y_train, y_cv = ms.train_test_split(X_train, y_train, test_size=0.25)
            return X_train, X_cv, X_test, y_train, y_cv, y_test
        else:
            if random_seed:
                X_train, X_test, y_train, y_test = ms.train_test_split(X_all, y_all, test_size=0.25, random_state=random_seed)
            else:
                X_train, X_test, y_train, y_test = ms.train_test_split(X_all, y_all, test_size=0.25)
            
            return X_train, X_test, y_train, y_test
            
    def get_cv(self):
        return self.cv
            
        
            
    def use_cv(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.feature_type = 'cv'
        self.use_binary = False
        self.cv = CountVectorizer(stop_words=self.remove_stopwords, binary=self.use_binary)
        
    def use_bcv(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.feature_type = 'cv'
        self.use_binary = True
        self.cv = CountVectorizer(stop_words=self.remove_stopwords, binary=self.use_binary)
        
    def use_tfidf(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.feature_type = 'tfidf'
        self.use_binary = False
        self.cv = CountVectorizer(stop_words=self.remove_stopwords, binary=self.use_binary)