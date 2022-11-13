import unittest
from Poisson_LDS import Poisson_LDS_Expected_Likelihood
import torch

class Poisson_LDS_Tests(unittest.TestCase):
    def test_Expected_likelihood_shape(self):
        xt_dim =5
        zt_dim = 2
        batch = 100
        time= 12
        model = Poisson_LDS_Expected_Likelihood(xt_dim,zt_dim)
        x = torch.randint(low = 0, high= 100,size =(time,xt_dim))
        z = torch.rand((batch,time,zt_dim))
        print(model(x,z))
