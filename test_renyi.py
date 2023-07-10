import unittest
import numpy as np
from renyi import *

class MyTestCase(unittest.TestCase):
    def test_gaussian(self):
        dif = []

        for _ in range(100):
            num_points = 1000
            p = np.random.random()
            mean = (0,0)
            conv = [[1,p],
                    [p,1]]

            points = np.random.multivariate_normal(mean,conv,size=num_points)

            ts = tau_s(points,lower_bounds=[-3],upper_bounds=[3])
            tst = tau_s_t(points,lower_bounds=[-3,-3],upper_bounds=[3,3],bins=10)

            metric_renyi = (tst - ts)/tst
            ref_p = 1 - (1 - p**2)**0.5

            dif.append(abs(metric_renyi - ref_p))

        print(np.mean(dif))
        print(dif)

        self.assertLess(np.mean(dif),0.15)

if __name__ == '__main__':
    unittest.main()
