# Core Learning Algorithms: Linear Regression, Classification, Clustering, Hidden Morkov Models

# Line of best fit refers to a line through a scatter plot of data points that best expresses the relationship between those points.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc 

import tensorflow as tf

# using titanic dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# .head() method from pandas will show us the first 5 items in our dataframe.
print(dftrain.head())
print(dfeval.head())


