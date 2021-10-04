# -*- coding: utf-8 -*-
#IMPORTS
import numpy as np
#SELF IMPORTS
import data.generator as generator
from lib.func import *

#DESCENT METHODS
def sgd(x,dF,lam=.001,eta=.001):
    """ Gradient Descent Method """
    return x - eta*dF

def prox(x,dF,lam=.001,eta=.001):
    """ Proximal Gradient Method """
    return (x-eta*dF)/(1+2*lam)

def shrink(x,dF,lam=.001,eta=.001):
    """ Shrinkage Operator for Proximal Gradient Method """
    return LASSO(x-eta*dF, lam) 

def accProx(x,x0,dF,lam=.001,eta=.001):
    """ Accelerated Proximal Gradient Descent """
    theta = .001 
    return (x-eta*dF+theta*(x-x0))/(1+2*lam)

def accShrink(x,x0,dF,lam=.001,eta=.001):
    """ Accelerated Shrinkage """
    theta = .001
    return LASSO(x-eta*dF+theta*(x-x0), lam)

def nestProx(x,x0,dF,lam=.001,eta=.001):
    """ Nesterov Proximal Gradient Descent """
    theta = (1-np.sqrt(eta))**2
    y = x + theta*(x-x0)
    return (y-eta*dF+theta*(x-x0))/(1+2*lam)

def nestShrink(x,x0,dF,lam=.001,eta=.001):
    """ Nesterov Shrinkage """
    theta = (1-np.sqrt(eta))**2
    return (x-eta*dF+theta*(x-x0))/(1+2*lam)

#DESCENT METHODS
def desc(init, A, b, reg=zeroReg, method=sgd, lam=.001, eta=.001, eps=1e-3):
    n = 1
    x0 = init
    x = x0
    opt = generator.coefs
    nfe = []
    dx = [np.linalg.norm(linReg(x0,A,b)-linReg(opt,A,b))]
    i=0
    while dx[-1] > eps and i<1000:
        if i%100 == 0: print(i)
        dF = gradLR(x0,A,b)+reg(x0,lam)
        x = method(x0,dF,lam=lam, eta=eta)
        dx = dx +[np.linalg.norm(linReg(x,A,b)-linReg(opt,A,b))]
        x0 = x
        if dx[-1] < 10**(-n):
            nfe = nfe+[i]
            i=0
            n=n+1
        i=i+1
    itt = np.arange(len(nfe))
    return itt, nfe, dx


def accel(init, A, b, method=accProx, lam=.001, eta=.001, eps=1e-3):
    n = 1
    x0 = init
    x = x0
    opt = generator.coefs
    nfe = []
    dx = [np.linalg.norm(linReg(x0,A,b)-linReg(opt,A,b))]
    i=0
    while dx[-1] > eps and i<1000:
        if i%100 == 0: print(i)
        dF = gradLR(x0,A,b)
        temp = method(x0,x,dF,lam=lam, eta=eta)
        x0 = x
        x = temp
        dx = dx +[np.linalg.norm(linReg(x,A,b)-linReg(opt,A,b))]
        if dx[-1] < 10**(-n):
            nfe = nfe+[i]
            i=0
            n=n+1
        i=i+1
    itt = np.arange(len(nfe))
    return itt, nfe, dx
