# -*- coding: utf-8 -*-

"""
Created on mon Jul 16 20:20:21 2020

@author: Pritam
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np

#Reducing features using principal components

digits = datasets.load_digits()

features = StandardScaler().fit_transform(digits.data)

pca = PCA(n_components = 0.99,whiten = True)

features_pca = pca.fit_transform(features)

print('Original number of features: {}'.format(features.shape[1]))
print('Reduced number of features: {}'.format(features_pca.shape[1]))

#Reducing Data when data is linearly inseperable



features, _ = make_circles(n_samples = 10000,random_state = 1, noise = 0.1,factor = 0.1)

kpca = KernelPCA(kernel = 'rbf',gamma = 15,n_components = 1)
features_kpca = kpca.fit_transform(features)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])

#Reducing Features by maximising class seperability

iris = datasets.load_iris()
features = iris.data
target = iris.target

lda = LinearDiscriminantAnalysis(n_components = 1)
features_lda = lda.fit(features,target).transform(features)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_lda.shape[1])

print(lda.explained_variance_ratio_)

lda = LinearDiscriminantAnalysis(n_components=None)

features_lda = lda.fit(features, target)
# Create array of explained variance ratios
lda_var_ratios = lda.explained_variance_ratio_

def select_n_components(var_ratio, goal_var: float) -> int:
# Set initial variance explained so far
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
        # Return the number of components
        return n_components
    
    
print(select_n_components(lda_var_ratios, 0.95))

#Reducinf features using matrix factorization

digits = datasets.load_digits()

features = digits.data

nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)


print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_nmf.shape[1])

#Reducing features on sparse data

digits = datasets.load_digits()

features = StandardScaler().fit_transform(digits.data)

features_sparse = csr_matrix(features)

tsvd = TruncatedSVD(n_components=10)

features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)

print("Original number of features:", features_sparse.shape[1])
print("Reduced number of features:", features_sparse_tsvd.shape[1])

tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)

tsvd_var_ratios = tsvd.explained_variance_ratio_

def select_n_components(var_ratio, goal_var):
    
    total_variance = 0.0
    
    n_components = 0

    for explained_variance in var_ratio:
        
        total_variance += explained_variance
        
        n_components += 1
        
        if total_variance >= goal_var:
            
            break
        
    return n_components

print(select_n_components(tsvd_var_ratios,0.95))