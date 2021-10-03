# -*- coding: utf-8 -*-
#IMPORTS
import numpy as np
#SELF IMPORTS
from lib.func import *

#DESCENT METHODS
def sgd(x,dF,eta=.001):
    """ Gradient Descent Method """
    return x - eta*dF

#GENERAL REGULARIZED DESCENT METHOD
def desc(init, A, b, reg=gradL2, method=sgd, lam=.001, eta=.1, eps=1e-3):
    n = 1
    x0 = init
    nfe = []
    dx = [1]
    i=0
    while dx[-1] > eps:
        dF = gradLR(x0,A,b)+reg(x0,lam)
        x = method(x0,dF,eta=.1)
        dx = dx +[np.linalg.norm(x-x0)]
        x0 = x
        if dx[-1] < 10**(-n):
            nfe = nfe+[i]
            i=0
            n=n+1
        i=i+1
    itt = np.arange(len(nfe))
    return itt, nfe, dx
