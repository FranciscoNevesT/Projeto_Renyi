import unittest
import numpy as np
from renyi import *

class MyTestCase(unittest.TestCase):
    def test_gaussian(self):
        dif = []

        for _ in range(10):
            num_points = 10000
            p = np.random.random()
            mean = (0,0)
            conv = [[1,p],
                    [p,1]]

            points = np.random.multivariate_normal(mean,conv,size=num_points)

            ts = tau_s(points)
            tst = tau_s_t(points,lower_bounds=[-3,-3],upper_bounds=[3,3],bins=10)

            metric_renyi = (tst - ts)/tst
            ref_p = 1 - (1 - p**2)**0.5

            dif.append(abs(metric_renyi - ref_p))

        self.assertLess(np.mean(dif),0.15)

    def test_3_dimensions(self):
        num_points_t = 100
        num_points = 10

        t = np.random.uniform(0,100,size=num_points_t)

        cov = np.array([[1, 0.8],
                        [0.8, 1]])

        points = []
        for i in t:
            pointsxy = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=(num_points))

            points = points + [list(j) + [i] for j in pointsxy]

        points = np.array(points)

        lower_bounds = list(np.min(points,axis=0))
        upper_bounds = list(np.max(points,axis=0))

        ts = tau_s(points)

        tst = tau_s_t(points, lower_bounds=lower_bounds, upper_bounds=upper_bounds,bins=20)

        metric_renyi = (tst - ts) / tst

        self.assertGreater(metric_renyi,0)


if __name__ == '__main__':
    unittest.main()
