import unittest
from predep import *
import numpy as np
import matplotlib.pyplot as plt
from renyi import tau_s,tau_s_t

class MyTestCase(unittest.TestCase):

    def test_gaussian(self):
        dif = []

        for _ in range(100):
            num_points = 1000
            p = np.random.uniform(-0.95,0.95)
            v= 1

            mean = (0,0)
            conv = [[v*v,p*v*v],
                    [p*v*v,v*v]]

            points = np.random.multivariate_normal(mean,conv,size=num_points)

            s = points[:,0:1]
            t = points[:,1:]

            ts = predep_s(s)

            tst = predep_s_t(s,t)

            metric_renyi = (tst - ts)/tst
            ref_p = 1 - (1 - p**2)**0.5

            dif.append(abs(metric_renyi - ref_p))


        print()
        print("Dif: {}".format(np.mean(dif)))

        self.assertLess(np.mean(dif),0.1)

    def test_linear(self):
        size = 1000

        x1,x2,x3,x4 = np.random.uniform(0,1,size = (4,size))

        x1 = x1.reshape(-1,1)
        x2 = x2.reshape(-1,1)
        x3 = x3.reshape(-1,1)
        x4 = x4.reshape(-1,1)

        s = x1 + 2*x2 + 4*x3 + 8*x4 + np.random.normal(0,scale=1,size= size).reshape(-1,1)

        t = np.concatenate([x1,x2,x3,x4],axis=1)

        predep_v = predep(s,t[:,:],n_clusters=100)

        self.assertGreaterEqual(predep_v, 0)



if __name__ == '__main__':
    unittest.main()
