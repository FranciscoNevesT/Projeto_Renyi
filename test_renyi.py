import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
from renyi import *

class MyTestCase(unittest.TestCase):
    def test_gaussian(self):
        dif = []

        for _ in range(100):
            num_points = 1000
            p = np.random.uniform(-1,1)

            mean = (0,0)
            conv = [[1,p],
                    [p,1]]

            points = np.random.multivariate_normal(mean,conv,size=num_points)

            ts = tau_s(points)
            tst = tau_s_t(points,lower_bounds=[-3,-3],upper_bounds=[3,3],bins=10)

            metric_renyi = (tst - ts)/tst
            ref_p = 1 - (1 - p**2)**0.5

            dif.append(abs(metric_renyi - ref_p))


        self.assertLess(np.mean(dif),0.1)

    def test_3_dimensions(self):
        num_points_t = 10
        num_points = 100

        t = np.random.uniform(0,100,size=num_points_t)

        cov = np.array([[1, 0],
                        [0, 1]])

        points = []
        for i in t:
            pointsxy = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=(num_points))

            points = points + [list(j) + [i] for j in pointsxy]

        points = np.array(points)

        lower_bounds = list(np.min(points,axis=0))
        upper_bounds = list(np.max(points,axis=0))

        ts = tau_s(points)

        tst = tau_s_t(points, lower_bounds=lower_bounds, upper_bounds=upper_bounds,bins=10)

        print(ts)
        print(tst)


        metric_renyi = (tst - ts) / tst

        print(metric_renyi)

        self.assertGreater(metric_renyi,0)


    def test_circule(self):
        num_points = 100000

        a = np.random.random(size=(num_points)) * 2*math.pi
        r = np.random.random(size=(num_points))

        r = r/4 + 3/4

        points = [[np.sin(a[i]) * r[i],np.cos(a[i]) * r[i]] for i in range(num_points)]
        points = np.array(points)

        plt.plot(points[:,0],points[:,1],"o")
        plt.show()

        lower_bounds = list(np.min(points,axis=0))
        upper_bounds = list(np.max(points,axis=0))

        ts = tau_s(points)

        tst = tau_s_t(points, lower_bounds=lower_bounds, upper_bounds=upper_bounds,bins=20)

        metric_renyi = (tst - ts) / tst

        print(ts)
        print(tst)


        metric_renyi = (tst - ts) / tst

        print(metric_renyi)


        self.assertGreater(metric_renyi,0)


if __name__ == '__main__':
    unittest.main()
