# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 20:29:58 2017

@author: Joost
"""

import numpy as np
from machine_learner import MachineLearner
from sklearn import svm, metrics
from sklearn.ensemble import ExtraTreesClassifier
from my_nn import NeuralNetwork


ml = MachineLearner()

nn_clf_lambda = lambda: NeuralNetwork( [ml.get_nr_of_features(), 100, 1], 
                        epochs=230, 
                        mini_batch_size=50, 
                        eta=0.5, 
                        lmbda=0.5,
                        debug=False)

ml.add_classifier("Neural Network", nn_clf_lambda)

svm_clf_lambda = lambda: svm.SVC()
ml.add_classifier("SVM", svm_clf_lambda)

etc_clf_lambda = lambda: ExtraTreesClassifier(n_estimators=450, 
                                              criterion='entropy', 
                                              bootstrap=True, 
                                              oob_score=True, 
                                              min_samples_split=13, 
                                              max_features=100,
                                              max_depth=None, 
                                              min_samples_leaf=1)

ml.add_classifier("Extra trees", etc_clf_lambda)

avs = ml.test_n_times(5, True)

#print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(y_test, y_test_pred)))
