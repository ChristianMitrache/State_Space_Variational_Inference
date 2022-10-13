import unittest
import torch
from Mean_Cov_Models import Inverse_Variance_Model

########################################################################################################

class Test_Variance_Model(unittest.TestCase):
    def test_variance_model_shapes(self):
        time = 100
        xt_dim = 10
        x = torch.rand((100,10))
        linear_layers = (7,2,7)
        non_linearity = torch.nn.ReLU()
        inverse_variance_model = Inverse_Variance_Model(xt_dim, linear_layers, non_linearity)
        L_D, L_B = inverse_variance_model(x)
        self.assertTrue(L_B.shape ==(99,10,10) and L_D.shape == (100,10,10))


if __name__ == '__main__':
    unittest.main()



if __name__ == '__main__':
    unittest.main()
