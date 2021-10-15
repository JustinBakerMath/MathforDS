# -*- coding: utf-8 -*-
# IMPORTS
import numpy as np

# OBJECTIVE FUNCTION
def linReg(x, A, b):
    """ Linear Regression Function
    f(x) = 1/(2n)\sum log(1+exp(-b_i a_i' x)) """
    n=len(b)
    F = 0
    for i in range(n):
        F = F+np.log(1+np.exp(-b[i]*(A[i]@x)))
    return F/(2*n)

def gradLR(x, A, b):
    """ Gradient of Linear Regression Function
    f(x) = 1/(2n)\sum [(-b_i)exp(-b_i a_i' x)a_i]/[1+exp(-b_i a_i' x)] """
    n=len(b)
    F = 0
    for i in range(n):
        F = F-b[i]*A[i]+b[i]*A[i]/(1+np.exp(-b[i]*(A[i]@x)))
    return F/(2*n)

# REGULARIZATION TERMS
def zeroReg(x,lam):
    """ No Regularization Term """
    return np.zeros(len(x))

def l2Reg(x,lam):
    """ L2 Regularization """
    return lam*np.linalg.norm(x,2)**2

def gradl2(x,lam):
    """ Gradient of L2 Regularization """
    return 2*lam*x

def l1Reg(x, lam):
    """ L1 Regularization """
    return lam*np.linalg.norm(x,1)

def gradl1(x,lam):
    """ Subgradient of L1 Regularization Term """
    return lam*np.sign(x)

def zeroGrad(x,lam):
    """ Subgradient of L1 Regularization Term """
    return np.zeros(len(x))

def LASSO(x,lam):
    """ LASSO projection of L1 Regularization """
    r = np.zeros(len(x))
    for i,xi in enumerate(x):
        if abs(xi) >= lam:
           r[i] = xi-lam*np.sign(xi)
    return r
