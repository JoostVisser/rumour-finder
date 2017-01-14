# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 00:02:38 2017

@author: Joost
"""
import numpy as np
from preprocessor import PreProcessor
import sklearn.model_selection as ms
from sklearn import svm, metrics


class MachineLearner:
    
    def __init__(self, feature_type='cv'):
        self.feature_type = feature_type
        self.pp = PreProcessor(feature_type)
        self.X, self.y = self.pp.get_features()
        self.classifiers = {}
        
    def add_classifier(self, name, clf_lambda):
        self.classifiers[name] = clf_lambda    
        
    def use_cv(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.pp.use_cv()
        self.X, self.y = self.pp.get_features()
        
    def use_tfidf(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.pp.use_tfidf()
        self.X, self.y = self.pp.get_features()
        
    def get_nr_of_features(self):
        """ Returns the number of features of X. In other words, the 
        vocabulary size.
        """
        return np.size(self.X, 1)
        
    def test_once(self, print_results=False, print_detailed_results=False):
        """
        Tests all the classifiers in the classifier-array and returns the 
        resulting training set, cross-validation set and test-set accuracy.
        These tests are generated randomly from the test set with a ratio
        of 60% test set, 20% cross-validation set and 20%
        """
        # Can specify a seed by setting random_state=number.
        X_train, X_test, y_train, y_test = ms.train_test_split(self.X, self.y, test_size=0.2)
        X_train, X_cv, y_train, y_cv = ms.train_test_split(X_train, y_train, test_size=0.25)
        
        classifier_scores = {}
        
        for name, clf_lambda in self.classifiers.items():
            print("Classifying " + name + " ...")
            # Creating the classifier
            classifier = clf_lambda()
            # Fitting the data
            classifier.fit(X_train, y_train)
            
            # Calculating the resulting scores
            y_train_pred = classifier.predict(X_train)
            y_cv_pred =  classifier.predict(X_cv)
            y_test_pred = classifier.predict(X_test)
            train_score = metrics.accuracy_score(y_train, y_train_pred)    
            cv_score = metrics.accuracy_score(y_cv, y_cv_pred)    
            test_score = metrics.accuracy_score(y_test, y_test_pred)    
        
            classifier_scores[name] = \
                np.array([train_score, cv_score, test_score])
                
        if print_results:
            self.print_scores(classifier_scores, 1)
        
        return classifier_scores

    def test_n_times(self, n, print_results=False):
        """
        Tests all the classifiers in the classifier-array n times and returns 
        an average accuracy for the three different test sets.
        These tests are generated randomly from the test set with a ratio
        of 60% test set, 20% cross-validation set and 20%
        """
        average_classifier_scores = {}
        
        for i in range(n):
            print("Round " + str(i+1) + " start")
            classifier_scores = self.test_once()
            if i == 0:
                average_classifier_scores = classifier_scores
            else:
                for clf_name, scores in classifier_scores.items():
                    average_classifier_scores[clf_name] += scores
            print("Round " + str(i+1) + " finished")
            
        # Taking the average by dividing the scores by N.
        for scores in average_classifier_scores.values():
            scores = scores / n
        
                
        if print_results:
            self.print_scores(classifier_scores, n)
            
        return average_classifier_scores
                    
    def print_scores(self, classifier_scores, n):
        print()
        print("------------------Scores------------------")
        print("Average of " + str(n) + " test(s)")
        print()
        for clf_name, scores in classifier_scores.items():
            print(clf_name + " results:")
            print("Training set accuracy: %s" % scores[0])
            print("Cross-validation set accuracy: %s" % scores[1])
            print("Test set accuracy: %s" % scores[2])
            print()
    
    