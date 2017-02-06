# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 02:39:01 2017

@author: Joost
"""

import numpy as np
from preprocessor import PreProcessor
from sklearn import metrics
from my_nn import NeuralNetwork
import data_reader as dr
import random

class Demo:
    
    def __init__(self, feature_type='bcv', remove_stopwords=False):
        self.feature_type = feature_type
        self.remove_stopwords=remove_stopwords
        self.pp = PreProcessor(feature_type, remove_stopwords)
        # X and y are kept track of for the use of get_nr_of_features()
        self.X, self.y = self.pp.get_features_and_labels()
        self.nn = NeuralNetwork( [np.size(self.X, 1), 100, 1], 
                                epochs=250, 
                                mini_batch_size=30, 
                                eta=0.3, 
                                lmbda=0.7,
                                debug=False)
        self.cv = self.pp.get_cv()
        self.trained = False
    
    def start_training(self):
        print("Start training the neural network... ")
        X_train, X_test, y_train, y_test = \
                self.pp.get_all_sets(cv=False)
        
        self.nn.fit(X_train, y_train)
        print("Training finished, here are the test scores:")
        # Calculating the resulting scores
        y_test_pred = self.nn.predict(X_test)        
        test_acc = metrics.accuracy_score(y_test, y_test_pred)
        test_f1 = metrics.f1_score(y_test, y_test_pred)  
        
        print("Test set accuracy: %s" % test_acc)
        print("Test set F1-score: %s" % test_f1)
        input("Press ENTER to continue")
        print()
        self.trained = True
    
    def start_demo(self):
        tweet_text, tweet_labels = dr.load_data()
        
        
        while(True):
            
            
            indices = random.sample(list(range(610)),  5)
#            tweet_1 = tweet_text[indices[0]]

            
            for i in range(5):
                print((str(i+1) + " = " + tweet_text[indices[i]]).encode('utf-8'))
                print()
            
            while(True):
                tweet_nr = input("Please input which tweet you want to check (type 0 to stop): ")
                if tweet_nr == "0" or tweet_nr == "1" or tweet_nr == "2" or tweet_nr == "3" or tweet_nr == "4" or tweet_nr == "5":
                    break
            
            
            if tweet_nr == "0":
                break
            
            tweet = tweet_text[indices[int(tweet_nr) - 1]]
            X_demo = np.array(self.cv.transform([tweet]).todense())
            y_pred = self.nn.predict(X_demo)
            answer = "Rumour" if y_pred[0] else "Non-rumour"
            print("Neural network predicts: " + answer)
            answer_act = "Rumour" if tweet_labels[int(tweet_nr)] else "Non-rumour"
            
            print("Actual label: " + answer_act)
            print()
            input("Press ENTER to continue")
            print()
            
            
            
            