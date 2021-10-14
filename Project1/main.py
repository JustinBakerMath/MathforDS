# -*- coding: utf-8 -*-
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np

# SELF IMPORTS
from data import generator 
from lib.func import *
from lib.opt import *
from lib.utils import *
from lib.vis import *

np.random.seed(0)

# DATA
A = generator.A
b = generator.b
opt = generator.coefs
init = np.random.random(50)
lam = .001
eta = .01

#SUB-GRADIENT
print("Subgradient L2")
dxL1 = desc(init, A, b, opt, lam=lam, eta=eta, reg=l2Reg)
print("Subgradient L1")
dxL2 = desc(init, A, b, opt, lam=lam, eta=eta, reg=LASSO)
#PROX-GRADIENT
print("Proximal gradient L2")
dxL3 = desc(init, A, b, opt, lam=lam, eta=eta, method=prox)
print("Proximal gradient L1")
dxL4 = desc(init, A, b, opt, lam=lam, eta=eta, method=shrink)
#HBALL-PROX
print("Heavyball proximal gradient L2")
dxL5 = accel(init, A, b, opt, lam=lam, eta=eta, method=hbProx)
print("Heavyball proximal gradient L1")
dxL6 = accel(init, A, b, opt, lam=lam, eta=eta, method=hbShrink)
#NEST-PROX
print("Nesterov proximal gradient L2")
dxL7 = accel(init, A, b, opt, lam=lam, eta=eta, method=nestProx)
print("Nesterov proximal gradient L1")
dxL8 = accel(init, A, b, opt, lam=lam, eta=eta, method=nestShrink)
#RESTART-NEST
print("Adaptive restart Nesterov proximal gradient L2")
dxL9 = accel(init, A, b, opt, lam=lam, eta=eta, method=nestProx, restart=True)
print("Adaptive restart Nesterov proximal gradient L1")
dxL10 = accel(init, A, b, opt, lam=lam, eta=eta, method=nestShrink, restart=True)

#PLOTTING
DX = [dxL1, dxL2, dxL3, dxL4, dxL5, dxL6, dxL7, dxL8, dxL9, dxL10]
np.savez('data/output', dxL1, dxL2, dxL3, dxL4, dxL5, dxL6, dxL7, dxL8, dxL9, dxL10)
colors = ['k','k--','r','r--','b','b--', 'g', 'g--', 'm', 'm--']
comparison(DX,colors)