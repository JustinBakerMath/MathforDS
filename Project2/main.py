import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../")

from lib.kmeans import *

X= -0.5 + np.random.rand(100,2)
X1 = 0.5 + np.random.rand(50,2)
X[50:100, :] = X1


kmeans = KMeans(X,4)

kmeans.train()
classes = kmeans.predict()