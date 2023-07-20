import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
from renyi import *

class MyTestCase(unittest.TestCase):
    def test_gaussian(self):
        dif = []

        ms = []
        rs = []

        tsts = []
        tss = []

        for _ in range(100):
            num_points = 1000
            p = np.random.uniform(-0.95,0.95)
            v= 1

            mean = (0,0)
            conv = [[v*v,p*v*v],
                    [p*v*v,v*v]]

            points = np.random.multivariate_normal(mean,conv,size=num_points)
            ts = tau_s(points,bounds_type="inf")

            tst = tau_s_t(points)

            metric_renyi = (tst - ts)/tst
            ref_p = 1 - (1 - p**2)**0.5

            tss.append(ts)
            tsts.append(tst)

            dif.append(abs(metric_renyi - ref_p))

            ms.append(p)
            rs.append(abs(metric_renyi - ref_p))

        plt.plot(ms,rs,"o")
        plt.show()

        print()
        print(np.mean(dif))
        print("Dif: {}".format(np.mean(dif)))
        print("Ts: {}|{}".format(np.mean(tss),np.std(tss)))
        print("Tst: {}|{}".format(np.mean(tsts),np.std(tsts)))

        self.assertLess(np.mean(dif),0.1)

    def test_3_dimensions(self):
        cov = np.array([[1, 0.8],
                        [0.8, 1]])

        num_points = 1000
        points = []

        for _ in range(num_points):
            t = np.random.uniform(0,100)
            xy = list(np.random.multivariate_normal(mean=(0, t), cov=cov))

            points.append(xy + [t])

        points = np.array(points)

        #plt.plot(points[:,0],points[:,1],"o")
        #plt.show()

        ts = tau_s(points)

        tst = tau_s_t(points)

        print(ts)
        print(tst)

        metric_renyi = (tst - ts) / tst

        print(metric_renyi)

        self.assertGreater(metric_renyi,0)


    def test_circule(self):
        num_points = 1000

        points = []
        while len(points) < num_points:
            x,y = np.random.uniform(-1,1,2)

            if x**2 + y**2 >= 3/4 and x**2 + y**2 <= 1:
                points.append([x,y])

        points = np.array(points)

        plt.plot(points[:,0],points[:,1],"o")
        plt.show()

        ts = tau_s(points)

        tst = tau_s_t(points)

        print(ts)
        print(tst)


        metric_renyi = (tst - ts) / tst

        print(metric_renyi)


        self.assertGreater(metric_renyi,0)


if __name__ == '__main__':
    unittest.main()
