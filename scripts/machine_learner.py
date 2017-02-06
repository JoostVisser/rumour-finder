# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 00:02:38 2017

@author: Joost
"""
import numpy as np
from preprocessor import PreProcessor
from sklearn import metrics


class MachineLearner:
    
    def __init__(self, feature_type='cv', remove_stopwords=False):
        self.feature_type = feature_type
        self.remove_stopwords=remove_stopwords
        self.pp = PreProcessor(feature_type, remove_stopwords)
        # X and y are kept track of for the use of get_nr_of_features()
        self.X, self.y = self.pp.get_features_and_labels()
        self.classifiers = {}
        
    def add_classifier(self, name, clf_lambda):
        self.classifiers[name] = clf_lambda    
        
    def use_cv(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.pp.use_cv()
        self.X, self.y = self.pp.get_features()
        
    def use_bcv(self):
        """
        Use tf-idf to convert all words into features.
        """
        self.pp.use_bcv()
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
        
    def test_once(self, print_results=False, random_seed=False):
        """
        Tests all the classifiers in the classifier-array and returns the 
        resulting training set, cross-validation set and test-set accuracy.
        These tests are generated randomly from the test set with a ratio
        of 60% test set, 20% cross-validation set and 20%
        """
        X_train, X_test, y_train, y_test = \
            self.pp.get_all_sets(cv=False, random_seed)
        
        classifier_scores = {}
        
        for name, clf_lambda in self.classifiers.items():
#            print("Classifying " + name + " ...")
            # Creating the classifier
            classifier = clf_lambda()
            # Fitting the data
            classifier.fit(X_train, y_train)
            
            # Calculating the resulting scores
            y_train_pred = classifier.predict(X_train)
#            y_cv_pred =  classifier.predict(X_cv)
            y_test_pred = classifier.predict(X_test)
            
            train_acc = metrics.accuracy_score(y_train, y_train_pred)    
#            cv_acc = metrics.accuracy_score(y_cv, y_cv_pred)    
            test_acc = metrics.accuracy_score(y_test, y_test_pred)    
            
            train_f1 = metrics.f1_score(y_train, y_train_pred)    
#            cv_f1 = metrics.f1_score(y_cv, y_cv_pred)    
            test_f1 = metrics.f1_score(y_test, y_test_pred)   
        
            classifier_scores[name] = \
                np.array([train_acc, train_f1, test_acc, test_f1])
                
        if print_results:
            self.print_scores(classifier_scores, 1)
        
        return classifier_scores

    def test_n_times(self, n, print_results=False, random_seed=False):
        """
        Tests all the classifiers in the classifier-array n times and returns 
        an average accuracy for the three different test sets.
        These tests are generated randomly from the test set with a ratio
        of 60% test set, 20% cross-validation set and 20%
        """
        average_classifier_scores = {}
        
        for i in range(n):
#            print("Round " + str(i+1) + " start")
            if random_seed:
                random_seed += 1 # Changing the seed for different sets.
            classifier_scores = self.test_once(random_seed=random_seed)
            if i == 0:
                average_classifier_scores = classifier_scores
            else:
                for clf_name, scores in classifier_scores.items():
                    average_classifier_scores[clf_name] += scores
#            print("Round " + str(i+1) + " finished")
            
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
            print("Training set F1-score: %s" % scores[1])
#            print("Cross-validation set accuracy: %s" % scores[2])
#            print("Cross-validation set F1-score: %s" % scores[3])
            print("Test set accuracy: %s" % scores[2])
            print("Test set F1-score: %s" % scores[3])
            print()
    
    