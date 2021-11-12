import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size':20})

sys.path.append("../")

from lib.kmeans import *

# Problem 3
mat = loadmat("./data/kmeansgaussian.mat")['X']

col = []
for seed in range(10):
    row = []
    for k in range(10):
        classes, cent, dist = kmeans(mat,k+1,seed=seed)
        m = measure(dist,classes)
        row = row + [m]
    col = col + [row]

arr = np.round(np.array(col),2)

df = pd.DataFrame(np.round(np.array(col),2))

colors = ['k--', 'brown', 'red', 'blue', 'green',
        'indigo', 'teal', 'darkred', 'darkblue', 'darkgreen']

plt.figure(tight_layout=True, dpi=80)
for i,row in enumerate(arr):
    plt.plot(row, colors[i])
plt.xlabel("k clusters")
plt.ylabel("Loss")
plt.savefig('./out/prob3.pdf')
plt.show()

df.to_csv('./out/prob3.csv')
print("Finished Problem 3")


# Problem 4
df = pd.read_csv('./data/iris.data.csv', names = ["idx","x1", "x2", "x3", "name"], index_col=False)
names = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
df["name"] = df["name"].map(names)
true = df["name"].to_numpy()
print(df.head())

iris = df[["x1","x2","x3"]].to_numpy()

row = []
for seed in range(10):
    classes, cent, dist = kmeans(iris,3,distance=cos,seed=seed)
    m = measure(dist,classes)
    row = row + [m]
    if seed == 9:
        predict = classes

out = pd.DataFrame(np.array(row))

out.to_csv('./out/prob4.csv')
print("Finished Problem 4")

fig = plt.figure(tight_layout=True, dpi=80)
ax1 = plt.subplot(121, projection='3d')
colors = {0:'red',1:'blue',2:'green'}
for i,iris_ in enumerate(iris):
    ax1.scatter(iris_[0],iris_[1],iris_[2], color=colors[df["name"][i]])
ax1.set_title("True Class Values")
ax2 = plt.subplot(122, projection='3d')
colors = {2:'red',1:'blue',0:'green'}
for i,iris_ in enumerate(iris):
    ax2.scatter(iris_[0],iris_[1],iris_[2], color=colors[predict[i]])
ax2.set_title("Predicted Class Values")

# Rearrange to predictions
true[true == 0] = 3
true[true==2]=0
true[true==3]=2

print(true, predict)

print(divergence(true,predict))

plt.savefig('./out/prob4.pdf')
plt.show()