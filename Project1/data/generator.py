# -*- coding: utf-8 -*-
"""
Data generation for logistic regression
"""
import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz

np.random.seed(0)

n_features = 50  # The dimension of the feature is set to 50
n_samples = 100 # Generate 1000 training data

idx = np.arange(n_features)
coefs = ((-1) ** idx) * np.exp(-idx/10.)
coefs[20:] = 0.


def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))

def sim_logistic_regression(coefs, n_samples=1000, corr=0.5):
    """"
    Simulation of a logistic regression model
    
    Parameters
    coefs: `numpy.array', shape(n_features,), coefficients of the model
    n_samples: `int', number of samples to simulate
    corr: `float', correlation of the features
    
    Returns
    A: `numpy.ndarray', shape(n_samples, n_features)
       Simulated features matrix. It samples of a centered Gaussian vector with covariance 
       given bu the Toeplitz matrix
    
    b: `numpy.array', shape(n_samples,), Simulated labels
    """
    cov = toeplitz(corr ** np.arange(0, n_features))
    A = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    p = sigmoid(A.dot(coefs))
    b = np.random.binomial(1, p, size=n_samples)
    b = 2 * b - 1
    return A, b

A, b = sim_logistic_regression(coefs, n_samples)

out = np.vstack((A.T,b))
np.savez('./data/data', out)