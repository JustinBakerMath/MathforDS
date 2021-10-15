# -*- coding: utf-8 -*-
#IMPORTS
import numpy as np
from tqdm import trange
#SELF IMPORTS
import data.generator as generator
from lib.func import *

#DESCENT METHODS
def sgd(x,dF,lam=.001,eta=.001):
    """ Gradient Descent Method """
    return x - eta*dF

def prox(x,dF,lam=.001,eta=.001):
    """ Proximal Gradient Method """
    return (x-eta*dF)/(1+lam)

def shrink(x,dF,lam=.001,eta=.001):
    """ Shrinkage Operator for Proximal Gradient Method """
    return LASSO(x-eta*dF,lam)

def hbProx(x,x0,dF,t=0,theta=.1,lam=.001,eta=.001):
    """ Heavy Ball Proximal Gradient Descent """
    return (x-eta*dF+theta*(x-x0))/(1+lam)

def hbShrink(x,x0,dF,t=0,theta=.1,lam=.001,eta=.001):
    """ Heavy Ball Shrinkage """
    return LASSO(x-eta*dF+theta*(x-x0), lam)

def nestProx(x,x0,dF,t,theta=0,lam=.001,eta=.001):
    """ Nesterov Proximal Gradient Descent """
    return (x-eta*dF)/(1+lam)

def nestShrink(x,x0,dF,t,theta=0,lam=.001,eta=.001):
    """ Nesterov Shrinkage """
    return LASSO(x-eta*dF, lam)

#DESCENT METHODS
def desc(init, A, b, fopt, method=sgd, reg=zeroReg, gradReg=zeroGrad, lam=.001, eta=.001, eps=1e-10, itrs=1000):
    x = init
    X = [init]
    df = [np.linalg.norm(linReg(x,A,b)+reg(x,lam)-fopt)]
    for t in trange(itrs):
        dF = gradLR(x,A,b)+gradReg(x,lam)
        x = method(x,dF,lam=lam, eta=eta)
        X = X+[x]
        df = df +[np.linalg.norm(linReg(x,A,b)+reg(x,lam)-fopt)]
        if df[-1]<eps:
            break
    print(x)
    return  np.array(X), np.array(df)

def accel(init, A, b, fopt, method=hbProx, reg=zeroReg, gradReg=zeroGrad, lam=.001, eta=.001, restart=False, nesterov = False, eps=1e-10, itrs=1000):
    x0 = init
    x = x0
    X = [init]
    df = [np.linalg.norm(linReg(x0,A,b)+reg(x0,lam)-fopt)]
    t = 1
    for _ in trange(itrs):
        if nesterov:
            y = x + t/(t+3)*(x-x0)
            dF = gradLR(y,A,b)+gradReg(y,lam)
            temp = method(y,x0,dF,t=t,lam=lam, eta=eta)
        else:
            dF = gradLR(x,A,b)+gradReg(x,lam)
            temp = method(x,x0,dF,t=t,lam=lam, eta=eta)
        x0 = x
        x = temp
        X = X+[x]
        df = df +[np.linalg.norm(linReg(x,A,b)+reg(x,lam)-fopt)]
        t +=1
        if restart and linReg(x,A,b)>linReg(x0,A,b):
            t = 1
            x=x0
        if df[-1]<eps:
            break
    print(x)
    return np.array(X), np.array(df)
