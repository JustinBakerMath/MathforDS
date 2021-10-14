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

def shrink(x,dF,lam=.001,eta=.001):
    """ Shrinkage Operator for Proximal Gradient Method """
    return LASSO(x-eta*dF, lam) 

def prox(x,dF,lam=.001,eta=.001):
    """ Proximal Gradient Method """
    return (x-eta*dF)/(1+2*lam)

def hbProx(x,x0,dF,t=0,theta=.01,lam=.001,eta=.001):
    """ Heavy Ball Proximal Gradient Descent """
    return (x-eta*dF+theta*(x-x0))/(1+2*lam)

def hbShrink(x,x0,dF,t=0,theta=.01,lam=.001,eta=.001):
    """ Heavy Ball Shrinkage """
    return LASSO(x-eta*dF+theta*(x-x0), lam)

def nestProx(x,x0,dF,t,theta=0,lam=.001,eta=.001):
    """ Nesterov Proximal Gradient Descent """
    y = x + t/(t+3)*(x-x0)
    return (y-eta*dF)/(1+2*lam)

def nestShrink(x,x0,dF,t,theta=0,lam=.001,eta=.001):
    """ Nesterov Shrinkage """
    y = x + t/(t+3)*(x-x0)
    return LASSO(y-eta*dF, lam)

#DESCENT METHODS
def desc(init, A, b, opt, reg=zeroReg, method=sgd, lam=.001, eta=.001, eps=1e-3):
    x = init
    dx = [np.linalg.norm(linReg(x,A,b)-linReg(opt,A,b))]
    for _ in trange(1000):
        dF = gradLR(x,A,b)+reg(x,lam)
        x = method(x,dF,lam=lam, eta=eta)
        dx = dx +[np.linalg.norm(linReg(x,A,b)-linReg(opt,A,b))]
        if dx[-1]<eps:
            break
    print(np.linalg.norm(x-opt))
    return np.array(dx)

def accel(init, A, b, opt, method=hbProx, lam=.001, eta=.001, eps=1e-3, restart=False):
    x0 = init
    x = x0
    dx = [np.linalg.norm(linReg(x0,A,b)-linReg(opt,A,b))]
    t = 1
    for _ in trange(1000):
        dF = gradLR(x,A,b)
        temp = method(x,x0,dF,t=t,lam=lam, eta=eta)
        x0 = x
        x = temp
        dx = dx +[np.linalg.norm(linReg(x,A,b)-linReg(opt,A,b))]
        t +=1
        if restart and dF@(x-x0)>0:
            t = 1
            x=x0
        if dx[-1]<eps:
            break
    print(np.linalg.norm(x-opt))
    return np.array(dx)
