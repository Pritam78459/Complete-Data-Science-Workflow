# -*- coding: utf-8 -*-

"""
Created on mon Jul 13 7:29:32 2020

@author: Pritam
"""

#Encoding Nominal Categorical Features.

import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

#One-Hot Encoding

one_hot = LabelBinarizer()

#print(one_hot.fit_transform(feature))

#print(one_hot.classes_)

#Reverse one-Hot Encoding

#print(one_hot.inverse_transform(one_hot.transform(feature))) 

#print(pd.get_dummies(feature[:,0]))

#Multiclass One-Hot encoding
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

one_hot_multiclass = MultiLabelBinarizer()

#print(one_hot_multiclass.fit_transform(multiclass_feature))

#print(one_hot_multiclass.classes_)

#Encoding Ordinal Categories Features

dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})

scale_mapper = {"Low":1,
                "Medium":2,
                "High":3}

#print(dataframe['Score'].replace(scale_mapper))

dataframe = pd.DataFrame({"Score": ["Low",
                                    "Low",
                                    "Medium",
                                    "Medium",
                                    "High",
                                    "Barely More Than Medium"]})

scale_mapper = {"Low":1,
                "Medium":2,
                "Barely More Than Medium": 3,
                "High":4}

#print(dataframe['Score'].replace(scale_mapper))

scale_mapper = {"Low":1,
                "Medium":2,
                "Barely More Than Medium": 2.1,
                "High":3}
#print(dataframe["Score"].replace(scale_mapper))

#Encoding Dictionary Features

data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]

dictvectorizer  = DictVectorizer(sparse = False)

features = dictvectorizer.fit_transform(data_dict)

#print(features)

feature_names = dictvectorizer.get_feature_names()

#print(feature_names)

doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

doc_word_counts = [doc_1_word_count,
                   doc_2_word_count,
                   doc_3_word_count,
                   doc_4_word_count]

#print(dictvectorizer.fit_transform(doc_word_counts))

#Imputing Missing Class Values

X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])

imputed_values = trained_model.predict(X_with_nan[:,1:])

X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))

#print(np.vstack((X_with_imputed, X)))

X_complete = np.vstack((X_with_nan, X))

imputer = Imputer(strategy='most_frequent', axis=0)

#print(imputer.fit_transform(X_complete))

#Handling Imbalanced Classes

iris = load_iris()

features = iris.data

target = iris.target

features = features[40:,:]
target = target[40:]

target = np.where((target == 0),0,1)

weights = {0: .9,1: 0.1}

#print(RandomForestClassifier(class_weight = weights))
#print(RandomForestClassifier(class_weight = 'balanced'))

i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

n_class0 = len(i_class0)
n_class1 = len(i_class1)

i_class1_downsampled = np.random.choice(i_class1,size = n_class0,replace = False)

#print(np.hstack((target[i_class0],target[i_class1_downsampled])))

#print(np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5])

i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

#print(np.concatenate((target[i_class0_upsampled], target[i_class1])))

#print(np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5])