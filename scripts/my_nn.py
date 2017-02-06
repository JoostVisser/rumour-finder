# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:36:50 2017

@author: Joost
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
class NeuralNetwork(BaseEstimator, ClassifierMixin):

    def __init__(self, sizes, 
                 epochs=200,
                 mini_batch_size=10, 
                 eta=0.1, 
                 lmbda=5.0, 
                 debug=False):
        """The "sizes" list contains the sizes of the neural network.
        Example: [80, 20, 30] will have 80 input neurons, 20 neurons in
        the hidden layers and 30 output neurons.
        It's possible to have more than 1 hidden layer, but not recommended.
        'Training data' should be a list of tuples (x, y) representing
        the inputs and the desired outputs.
        'Epochs' are the number of times we should go over the training data.
        'Eta' is the learning rate.
        'mini_batch_size' is the size of the mini-batch.
        'lmbda' is lambda, i.e. the regularization parameter
        'debug' is True will let the neural network print some debug text.
        """

        self.num_layers = len(sizes)    # Number of layers
        self.sizes = sizes              # Sizes of the neural network layers
        # If the last layer has more than 1 neuron, than it's a multi-classifier.
        self.multi_classifier = sizes[-1] != 1

        # Initializing the weights and biases with a squashed Gaussian
        # Distribution for the weights.
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
        # Setting all variables
        self.epochs = epochs
        self.automated_epoch = False
        self.mb_size = mini_batch_size
        self.eta = eta
        self.lmbda = lmbda
        self.debug = debug

    def fit(self, X, y, X_cv = np.array([]), y_cv = np.array([])):
        """
        Fits the data 
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        if X_cv.any() and y_cv.any():
            size_X_cv = np.size(X_cv, 0)
            size_y_cv = np.size(y_cv, 0)
            if size_X_cv != size_y_cv:
                raise Exception("X and y of cross-validation set \
                                 are not of the same length")
            n_cv = size_X_cv
        elif self.automated_epoch:
            raise Exception("Need a cross-validation set for the automated\
                            epochs.")
        
        n = np.size(X, 0)
        
        # Setting some variables for automated epochs.
                    
        if self.automated_epoch:
            best_accuracy = 0.0
            best_in_n_count = 0
        
        for j in range(self.epochs):
            # Mini-batch stochastic gradient descent works by first shuffling
            # the dataset so different variations will happen each time.
            X, y = shuffle(X, y, random_state=0)
            
            # Putting the X and y pairs in mini_batches
            X_batches = [X[k:k+self.mb_size] 
                         for k in range(0, n, self.mb_size)]
            y_batches = [y[k:k+self.mb_size] 
                         for k in range(0, n, self.mb_size)]
                         
            for X_batch, y_batch in list(zip(X_batches, y_batches)):
                self.update_mini_batch(X_batch, y_batch, n)
                
            if self.debug:
                print("Epoch " + str(j) + " training complete")
            
            # Prints out result on the evaluation data if checked.
            if X_cv.any() and y_cv.any():
                accuracy = self.accuracy(X_cv, y_cv)
                if self.debug:
                    print(("Accuracy on evaluation data: {} / {}".format(
                        accuracy, n_cv)))

                # Code for checking if there is any improvement
                # in the last 10 epochs
                if self.automated_epoch:
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_in_n_count = 0
                    else:
                        best_in_n_count += 1

                    if best_in_n_count == self.stop_after_n:
                        if self.debug:
                            print("Reached best in 10 validation performance")
                        break

    def predict(self, X):
        """ Returns the values that the Neural Network predicts, given X.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)


        X = np.transpose(X)
        if self.multi_classifier:
            y_pred = np.argmax(self.feedforward(X), axis=0)
        else:
            y_pred = [0 if x < 0.5 else 1 for x in self.feedforward(X)[0]]
        return y_pred
        
    def stop_after_n(self, n, X_cv, y_cv):
        """Uses the cross-validation set to automatically determine when to stop learning.
        If there is no improvement in the accuracy, then the network will stop learning
        and return the best values.
        """
        self.stop_after_n = n    
        self.epochs = 10000
        self.automated_epoch = True        
        
    def feedforward(self, X):
        """Given a input matrix X, returns the output via feedforwarding
        through the neural network
        Notice that x is also the activation of the first layer"""
        a = X
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a
        
    def update_mini_batch(self, X, y, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.
        'mini-batch' is a list of tuples (x, y)
        'n' is the size of the training set, needed for L2 regularization.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        X = np.transpose(X)
        if self.multi_classifier:
            Y = self.vectorized_result(y)
        else:
            Y = y

        delta_nabla_b, delta_nabla_w = self.backprop(X, Y)
        
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # List of e.g. 3 of a matrix that contains all the current values.
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1-self.eta*(self.lmbda/n))*w-(self.eta/self.mb_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.eta/self.mb_size)*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, X, Y):
        """Performs feedforward and backpropagation.
        Input: a matrix with training input X and target output Y.
        Each column vector in X and Y represent one training case.

        Returns a tuple (delta_nabla_b, delta_nabla_w) which represents the
        gradient for the cost function over the bias and weights respectively.
        If there is more than one training case, then the delta_nabla_b and
        delta_nabla_v will contain the sum of the gradients of the training set

        """
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward, while storing the z and activations on the go.
        activation = X
        activations = [X]   # List of all activations for each layer.
        zs = []             # List of all z vectors for each layer.
        # Goes through all layers
        for b, w in zip(self.biases, self.weights):
            # Uses the fact that b will duplicate itself for each training case.
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward propagation
        delta = self.ce_delta(activations[-1], Y)

        # Store the current gradients w.r.t. bias and weights.
        delta_nabla_b[-1] = delta.sum(1)[:, np.newaxis]
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Go backwards through all layers up and until layer 2.
        for l in range(2, self.num_layers):
            # Perform backwards computation.
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # Store the current gradients w.r.t. bias and weights.
            delta_nabla_b[-l] = np.sum(delta, 1)[:, np.newaxis]
            delta_nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return delta_nabla_b, delta_nabla_w
        
    #### Miscellaneous functions
    def vectorized_result(self, y):
        """Return a n-dimensional matrix with a 1.0 in the j'th position
        and zeroes elsewhere.  This is used to convert a digit (0...9)
        into a corresponding desired output from the neural network.
    
        """
        e_idx = np.arange(len(y))
        e = np.zeros([self.sizes[-1], len(y)])
        e[y, e_idx] = 1.0
        return e

    def ce_cost(self, a, y):
        """Return the Cross-Entropy cost associated with an
        output 'a' and target output 'y'.
        The np.nan_to_num is there for stability reasons.
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def ce_delta(self, a, y):
        """Return the cross entropy error delta from the output layer. """
        return a - y

    def accuracy(self, X_test, y_test):
        """Return the number of inputs in for which the neural
        network outputs the correct result.
        The neural network's output is assumed to be the index
        of whichever neuron in the final layer has
        the highest activation.

        """
        # Predicted values with the given X's.
        y_pred = self.predict(X_test)

        # Sum of which the prediction is the same as the target output.
        return sum(y_pred == y_test)
        
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))