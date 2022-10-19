import unittest
import torch
from Mean_Cov_Models import Inverse_Variance_Model

########################################################################################################

class Test_Variance_Model(unittest.TestCase):
    def test_variance_model_shapes_no_batch(self):
        time = 100
        xt_dim = 40
        zt_dim =10
        x = torch.rand((time,xt_dim))
        linear_layers = (50,70,90)
        non_linearity = torch.nn.ReLU()
        inverse_variance_model = Inverse_Variance_Model(xt_dim,zt_dim, linear_layers,linear_layers, non_linearity)
        L_D, L_B = inverse_variance_model(x)
        self.assertTrue(L_B.shape ==(time-1,xt_dim,xt_dim) and L_D.shape == (time,xt_dim,xt_dim))


if __name__ == '__main__':
    unittest.main()



if __name__ == '__main__':
    unittest.main()
