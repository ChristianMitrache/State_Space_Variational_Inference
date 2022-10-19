import torch
from torch import nn
from torch.distributions import Poisson, MultivariateNormal

class Poisson_LDS_Expected_Likelihood(nn.Module):
    """
    Class that initializes Poisson LDS likelihood
    """
    def __init__(self, epsilon_precision: torch.Tensor,LDS_matrix: torch.Tensor,
                 loading_matrix: torch.Tensor,loading_bias: torch.Tensor):
        """
        Specifying class that takes in model parameters and feeds the corresponding dictionary to Super-Class
        In the notation of https://arxiv.org/pdf/1511.07367.pdf
        :param epsilon_precision: Q^-1
        :param LDS_matrix: A
        :param loading_matrix: C NOTE: C must be z_t x x_t matrix
        :param loading_bias: d
        """
        super().__init__()
        self.params = nn.ParameterDict({'epsilon_precision': epsilon_precision,
                                        'LDS_mat': LDS_matrix,
                                        'loading_mat':loading_matrix,
                                        'loading_bias':loading_bias})

    def forward(self,x:torch.Tensor,z:torch.Tensor):
        """
        This method must be
        :param x: a batch x Tx dim_x vector corresponding to the observations of the time series
        :param z: a single sample of size T x dim_x corresponding to a sample drawn from the approximate posterior.
        :return: Expected Log Joint Likelihood: P_{\theta}(x,z)
        """
        MultivariateNormal(loc = 0,precision_matrix=self.params['epsilon_precision']).log_prob(z[0,:,:]) + \
        torch.sum(MultivariateNormal(loc = z@self.params['LDS_mat'].T,
                                     precision_matrix=self.params['epsilon_precision']).log_prob(z[1:,:,:])) + \
        torch.sum(Poisson(rate = (z@self.params['loading_matrix'] + self.params['loading_bias'])).log_prob(x))
