from renyi import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

vs = np.linspace(0.1,100,num = 100)
p = 0.8
num_points = 10000
mean = (0, 0)

ref_p = 1 - (1 - p ** 2) ** 0.5

data = []
for v in vs:
    conv = [[v * v, p * v * v],
            [p * v * v, v * v]]

    points = np.random.multivariate_normal(mean, conv, size=num_points)

    metric_renyi = calc_renyi(points, num_threads=1,bandwidth="ISJ")

    if metric_renyi < 0:
        metric_renyi = 0.0

    data.append([v,metric_renyi])

    print("v: {}".format(v))

data = pd.DataFrame(data)

sns.lineplot(data = data, x = 0, y = 1)
plt.hlines(y= ref_p,xmin=0,xmax=100)


plt.show()


print(data)