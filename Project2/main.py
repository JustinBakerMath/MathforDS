import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys

sys.path.append("../")

from lib.kmeans import *

def measure(dist,classes):
    s = 0
    for i,d in enumerate(dist.T):
        class_ = d[classes == i]
        s = s+ sum(class_)
    return s/len(classes)

# X= -0.5 + np.random.rand(100,2)
# X1 = 0.5 + np.random.rand(50,2)
# X[50:100, :] = X1


# classes, cent, dist = kmeans(X,2,distance=eucl)

# plt.scatter(X[classes == 0, 0], X[classes == 0, 1], s = 20, c = 'b')
# plt.scatter(X[classes == 1, 0], X[classes == 1, 1], s = 20, c = 'r') 
# # plt.show()


# mat = loadmat("./data/kmeansgaussian.mat")['X']

# col = []
# for seed in range(10):
#     row = []
#     for k in range(10):
#         classes, cent, dist = kmeans(mat,k+1,seed=seed)
#         m = measure(dist,classes)
#         row = row + [m]
#     col = col + [row]

# df = pd.DataFrame(np.array(col))

# df.to_csv('./out/prob3.csv')
# print("Finished Problem 3")



df = pd.read_csv('./data/iris.data.csv', names = ["idx","x1", "x2", "x3", "name"], index_col=False)
names = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
df["name"] = df["name"].map(names)
print(df.head())

iris = df[["x1","x2","x3"]].to_numpy()

row = []
for seed in range(10):
    classes, cent, dist = kmeans(iris,3,distance=cos,seed=seed)
    m = measure(dist,classes)
    row = row + [m]
    if seed == 9:
        predict = classes

print(predict)
out = pd.DataFrame(np.array(row))

out.to_csv('./out/prob4.csv')
print("Finished Problem 4")

fig = plt.figure()
ax1 = plt.subplot(121, projection='3d')
colors = {0:'red',1:'blue',2:'green'}
for i,iris_ in enumerate(iris):
    ax1.scatter(iris_[0],iris_[1],iris_[2], color=colors[df["name"][i]])
ax2 = plt.subplot(122, projection='3d')
colors = {0:'red',1:'blue',2:'green'}
for i,iris_ in enumerate(iris):
    ax2.scatter(iris_[0],iris_[1],iris_[2], color=colors[predict[i]])

plt.show()