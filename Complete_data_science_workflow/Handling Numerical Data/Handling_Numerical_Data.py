# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 20:54:59 2020

@author: Pritam
"""

import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from fancyimpute import KNN


#Rescaling a feature

feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

minmax_scale = preprocessing.MinMaxScaler(feature_range = (0,1))
scaled_feature = minmax_scale.fit_transform(feature)

#print(scaled_feature)

#Standardising a feature

x = np.array([[-100.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

scaler = preprocessing.StandardScaler()

standardized = scaler.fit_transform(x)

#print(standardized)

#Robust Scaler

robust_scaler = preprocessing.RobustScaler()

#print(robust_scaler.fit_transform(x))

#Normalizing observations

features = np.array([[0.5,0.5],
            [1.1,3.4],
            [1.5,20.2],
            [1.63,34.4],
            [10.9,3.3]])

normalizer = preprocessing.Normalizer(norm ='l2')

#print(normalizer.fit_transform(features))

#Generating polynomial and interaction features.

features = np.array([[2,3],
                     [2,3],
                     [2,3]])

polynomial_interaction = preprocessing.PolynomialFeatures(degree = 2,include_bias =False)

#print(polynomial_interaction.fit_transform(features))

interaction = preprocessing.PolynomialFeatures(degree=2,
                interaction_only=True, include_bias=False)
#print(interaction.fit_transform(features))

#Transforming Features.

features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

def add_ten(x):
    return x + 10

ten_transformer = preprocessing.FunctionTransformer(add_ten)

#print(ten_transformer.transform(features))

#using pandas

df = pd.DataFrame(features, columns = ['feature_1','feature-2'])
#print(df.apply(add_ten))

#Detecting Outliers

features, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)

features[0,0] = 10000
features[0,1] = 10000

outlier_detector = EllipticEnvelope(contamination = .1)

outlier_detector.fit(features)

outlier_detector.predict(features)

feature = features[:,0]

def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

#print(indicies_of_outliers(feature))

#Handling Outliers
    
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

houses[houses['Bathrooms'] < 20]

houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)

houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

#Discretizating Features

age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])
    
binarizer = preprocessing.Binarizer(18)

#print(binarizer.fit_transform(age))

#print(np.digitize(age, bins = [20,30,64]))

#Grouping Observing Using Clustering

features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])


clusterer = KMeans(3, random_state=0)


clusterer.fit(features)

dataframe["group"] = clusterer.predict(features)

#print(dataframe.head())

#Deleting Observations with Missing Values

features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])
    
#print(features[~np.isnan(features).any(axis=1)])
    
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

#print(dataframe.dropna())

features, _ = make_blobs(n_samples = 1000,
                         n_features = 2,
                         random_state = 1)

scaler = preprocessing.StandardScaler()
standardized_features = scaler.fit_transform(features)

true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan

features_knn_imputed = KNN(k=5, verbose=0).complete(standardized_features)

#print("True Value:", true_value)
#print("Imputed Value:", features_knn_imputed[0,0])

mean_imputer = preprocessing.Imputer(strategy="mean", axis=0)

features_mean_imputed = mean_imputer.fit_transform(features)

print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])

