# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 20:29:58 2017

The different tests i 

@author: Joost
"""
from machine_learner import MachineLearner
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from my_nn import NeuralNetwork
from sklearn.ensemble import VotingClassifier

def classifiers_test():
    """
    Test for comparing the different data.
    """
    ml = MachineLearner()

    nn_clf_lambda = lambda: NeuralNetwork( [ml.get_nr_of_features(), 100, 1], 
                            epochs=250, 
                            mini_batch_size=30, 
                            eta=0.3, 
                            lmbda=0.7,
                            debug=False)
    ml.add_classifier("Neural Network", nn_clf_lambda)
    
    svm_clf_lambda = lambda: SVC()
    ml.add_classifier("SVM", svm_clf_lambda)
    
    etc_clf_lambda = lambda: RandomForestClassifier()
    ml.add_classifier("Random trees", etc_clf_lambda)
    
    gaus_clf_lambda = lambda: GaussianNB()
    ml.add_classifier("Gaussian", gaus_clf_lambda)
    
    tree_clf_lambda = lambda: DecisionTreeClassifier()
    ml.add_classifier("Decision tree", tree_clf_lambda)
    
    eclf = lambda: VotingClassifier(estimators=[('gaus', GaussianNB()), 
                                                ('rfc', RandomForestClassifier()), 
                                                ('dtc', DecisionTreeClassifier())], 
                                                voting='hard')
    
    ml.add_classifier("Voting classifier", eclf)
    ml.test_n_times(5, True)
    

def preprocessor_test():
    """
    Test for comparing the different ways of preprocessing the data.
    """
    
    ml1 = MachineLearner(feature_type='cv', remove_stopwords=False)
    ml2 = MachineLearner(feature_type='bcv', remove_stopwords=False)
    ml3 = MachineLearner(feature_type='tfidf', remove_stopwords=False)
    ml4 = MachineLearner(feature_type='cv', remove_stopwords=True)
    ml5 = MachineLearner(feature_type='bcv', remove_stopwords=True)
    ml6 = MachineLearner(feature_type='tfidf', remove_stopwords=True)
    
    nn1_clf_lambda = lambda: NeuralNetwork( [ml1.get_nr_of_features(), 100, 1], 
                            epochs=250, 
                            mini_batch_size=30, 
                            eta=0.3, 
                            lmbda=0.7,
                            debug=False)
    
    nn2_clf_lambda = lambda: NeuralNetwork( [ml2.get_nr_of_features(), 100, 1], 
                            epochs=250, 
                            mini_batch_size=30, 
                            eta=0.3, 
                            lmbda=0.7,
                            debug=False)
    
    nn3_clf_lambda = lambda: NeuralNetwork( [ml3.get_nr_of_features(), 100, 1], 
                            epochs=250, 
                            mini_batch_size=30, 
                            eta=0.3, 
                            lmbda=0.7,
                            debug=False)
    
    nn4_clf_lambda = lambda: NeuralNetwork( [ml4.get_nr_of_features(), 100, 1], 
                            epochs=250, 
                            mini_batch_size=30, 
                            eta=0.3, 
                            lmbda=0.7,
                            debug=False)
    
    nn5_clf_lambda = lambda: NeuralNetwork( [ml5.get_nr_of_features(), 100, 1], 
                            epochs=250, 
                            mini_batch_size=30, 
                            eta=0.3, 
                            lmbda=0.7,
                            debug=False)
    
    nn6_clf_lambda = lambda: NeuralNetwork( [ml6.get_nr_of_features(), 100, 1], 
                            epochs=250, 
                            mini_batch_size=30, 
                            eta=0.3, 
                            lmbda=0.7,
                            debug=False)
    
    ml1.add_classifier("Neural Network", nn1_clf_lambda)
    ml2.add_classifier("Neural Network", nn2_clf_lambda)
    ml3.add_classifier("Neural Network", nn3_clf_lambda)
    ml4.add_classifier("Neural Network", nn4_clf_lambda)
    ml5.add_classifier("Neural Network", nn5_clf_lambda)
    ml6.add_classifier("Neural Network", nn6_clf_lambda)
    
    # These all have the same random_seed to get the same test-cases.
    print()
    print("Testing CV, no stop words removed.")
    ml1.test_n_times(5, True, random_seed=144)
    
    print()
    print("Testing BCV, no stop words removed.")
    ml2.test_n_times(5, True, random_seed=144)
    
    print()
    print("Testing TF-IDF, no stop words removed.")
    ml3.test_n_times(5, True, random_seed=144)
    
    print()
    print("Testing CV, stop words are being removed.")
    ml4.test_n_times(5, True, random_seed=144)
    
    print()
    print("Testing BCV, stop words are being removed.")
    ml5.test_n_times(5, True, random_seed=144)
    
    print()
    print("Testing TF-IDF, stop words are being removed.")
    ml6.test_n_times(5, True, random_seed=144)
    