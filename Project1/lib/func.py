# -*- coding: utf-8 -*-
# IMPORTS
import numpy as np

# OBJECTIVE FUNCTION
def linReg(x, A, b):
    """ Linear Regression Function
    f(x) = 1/(2n)\sum log(1+exp(-b_i a_i' x)) """
    F = 0
    for i in range(len(b)):
        F = F+np.log(1+np.exp(-b[i]*np.dot(A[i],x)))
    return F/(2*len(b))

def gradLR(x, A, b):
    """ Gradient of Linear Regression Function
    f(x) = 1/(2n)\sum [(-b_i)exp(-b_i a_i' x)a_i]/[1+exp(-b_i a_i' x)] """
    F = 0
    for i in range(0,len(b)):
        F = F-1/2/len(b)*np.exp(-b[i]*np.dot(A[i],x))*A[i]*b[i]\
            /(1+np.exp(-b[i]*np.dot(A[i],x)))
    return F

# REGULARIZATION TERMS
def l2Reg(x,lam):
    """ L2 Regularization """
    return lam*np.linalg.norm(x,2)**2

def gradL2(x,lam):
    """ Gradient of L2 Regularization """
    return 2*lam*x

def l1Reg(x):
    """ L1 Regularization """
    return lam*np.linalg.norm(x,1)

def LASSO(x,lam):
    """ LASS projection of L1 Regularization """
    r = np.zeros(len(x))
    for i,xi in enumerate(x):
        if abs(xi) > lam:
           r[i] = xi-lam*np.sign(xi) 
    return r
