# -*- coding: utf-8 -*-

"""
Created on mon Jul 17 19:04:41 2020

@author: Pritam
"""

#Thresholding numerical feature variance

from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectPercentile
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model


iris = datasets.load_iris()

features = iris.data
target = iris.target

thresholder = VarianceThreshold(threshold = .5)

features_high_variance = thresholder.fit_transform(features)

print(features_high_variance[0:3])

thresholder.fit(features).variances_

#Never Standardize the data
scaler = StandardScaler()
features_std = scaler.fit_transform(features)


selector = VarianceThreshold()
print(selector.fit(features_std).variances_)

#Thresholding binary feature variance

features = [[0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]]

thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
print(thresholder.fit_transform(features))

#Handling Highly correlated features

features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])

dataframe = pd.DataFrame(features)

corr_matrix = dataframe.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(dataframe.drop(dataframe.columns[to_drop], axis=1).head(3))
print(dataframe.corr())

#Removing irrelevant features for classification

iris = load_iris()
features = iris.data
target = iris.target
#For categorical

features = features.astype(int)

chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

#For quantitative

fvalue_selector = SelectKBest(f_classif, k=2)
features_kbest = fvalue_selector.fit_transform(features, target)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

#For top n features

fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

#Recursively Eliminating Features

warnings.filterwarnings(action="ignore", module="scipy",
message="^internal gelsd")

features, target = make_regression(n_samples = 10000,
                                   n_features = 100,
                                   n_informative = 2,
                                   random_state = 1)

ols = linear_model.LinearRegression()

rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
print(rfecv.transform(features))

print(rfecv.n_features_)

print(rfecv.support_)

print(rfecv.ranking_)