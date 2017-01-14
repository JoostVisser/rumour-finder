# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:46:01 2017

@author: Joost
"""

from my_nn import NeuralNetwork


# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
import sklearn.model_selection as ms

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Create a classifier: a support vector classifier
classifier = NeuralNetwork([np.size(X, 1), 100, 10], 
                           epochs=0, 
                           mini_batch_size=20, 
                           eta=0.1, 
                           lmbda = 1)

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)
X_train, X_cv, y_train, y_cv = ms.train_test_split(X_train, y_train, test_size=0.25)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train, X_cv, y_cv)

# Now predict the value of the digit on the second half:
expected = y_test
predicted = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(X_test.reshape(len(X_test), 8, 8), predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()