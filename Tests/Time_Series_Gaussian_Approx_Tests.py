import unittest
import torch
from Mean_Cov_Models import Inverse_Variance_Model, Fully_Connected_Mean_Model
from Approximate_Posterior import Time_Series_Approx_Gaussian_First_Param

"""
This file tests the classes under the approximate posterior class.
"""

class Test_Time_Series_Approx_Gaussian_First_Param(unittest.TestCase):
    """
    This tests the implementation of the Time_Series_Approx_Gaussian_First_Param.
    """

    def test_forward_pass_dimensions(self):
        time = 100
        xt_dim = 10
        x = torch.rand((time,xt_dim))
        linear_layers = (7,2,7)
        non_linearity = torch.nn.ReLU()
        inverse_variance_model = Inverse_Variance_Model(xt_dim, linear_layers, non_linearity)
        mean_model = Fully_Connected_Mean_Model(xt_dim,linear_layers,non_linearity)
        approx_inference_object = Time_Series_Approx_Gaussian_First_Param(mean_model,inverse_variance_model)
        print(approx_inference_object(x).shape)
        self.assertTrue(approx_inference_object(x).shape == torch.zeros(100).shape)

    def test_batch_sample_size(self):
        time = 100
        xt_dim = 10
        batch= 100
        x = torch.rand((time,xt_dim))
        linear_layers = (7,2,7)
        non_linearity = torch.nn.ReLU()
        inverse_variance_model = Inverse_Variance_Model(xt_dim, linear_layers, non_linearity)
        mean_model = Fully_Connected_Mean_Model(xt_dim,linear_layers,non_linearity)
        approx_inference_object = Time_Series_Approx_Gaussian_First_Param(mean_model, inverse_variance_model)
        approx_inference_object(x)
        self.assertTrue(tuple(approx_inference_object.Sample(batch,x).shape) == (batch,time,xt_dim))

if __name__ == '__main__':
    unittest.main()
