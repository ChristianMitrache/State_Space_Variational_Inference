from typing import Optional

import torch
from torch import nn
from torch.distributions import Poisson, MultivariateNormal

def sample_from_Poisson_LDS(epsilon_precision: torch.Tensor, LDS_matrix: torch.Tensor,
                            loading_matrix: torch.Tensor, loading_bias: torch.Tensor, time_steps: int):
    """
    Function that produces sequence of observations x from sampling from a poisson LDS given the true parameters.
    In the notation of https://arxiv.org/pdf/1511.07367.pdf
    :param epsilon_precision: Q^-1
    :param LDS_matrix: A
    :param loading_matrix: C NOTE: C must be z_t x x_t matrix
    :param loading_bias: d
    :param time_steps: number of time steps
    :return: tensor of samples in shape: Tx x_dim

    Note: Batching for this function is not implemented.
    """
    # sampling from epsilon process.
    sampled_epsilon = MultivariateNormal(loc=torch.zeros(time_steps,epsilon_precision.shape[0]),
                                         precision_matrix=epsilon_precision).sample((time_steps,epsilon_precision.shape[0]))

    #initial sample assuming that z_0 = epsilon_0
    sample_tensor = torch.zeros((time_steps,loading_matrix.shape[0]))
    z_curr = sampled_epsilon[0,:]
    sample_tensor[0,:] = Poisson(rate = torch.exp(loading_matrix@z_curr+loading_bias))

    # Iterative sampling assuming z_t = Az_{t-1} + epsilon_t
    for i in range(1,time_steps):
        z_curr =  LDS_matrix@z_curr + sampled_epsilon[i,:]
        sample_tensor[i,:] = Poisson(rate = torch.exp(loading_matrix@z_curr+loading_bias))
    return sample_tensor


class Poisson_LDS_Expected_Likelihood(nn.Module):
    """
    Class that initializes Poisson LDS likelihood
    """
    def __init__(self, xt_dim,zt_dim, epsilon_precision = None, LDS_matrix= None, loading_matrix= None, loading_bias = None):
        """
        Specifying class that takes in model parameters and feeds the corresponding dictionary to Super-Class
        In the notation of https://arxiv.org/pdf/1511.07367.pdf
        :param xt_dim: dimension of observational time series
        :param zt_dim: dimension of latent space for each time-point.
        :param epsilon_precision: Q^-1
        :param LDS_matrix: A
        :param loading_matrix: C NOTE: C must be z_t x x_t matrix
        :param loading_bias: d
        """
        super().__init__()

        # Code Below checks each parameter and initializes randomly it if it hasn't been passed in by the user
        self.xt_dim = xt_dim
        self.zt_dim = zt_dim
        # epsilon covariance:
        if epsilon_precision is None:
            # Making this positive semi-definite
            epsilon_precision = torch.rand((zt_dim,zt_dim))
            epsilon_precision = epsilon_precision @ epsilon_precision.T
            # Exponentiating diagonal entries to make it positive definite
            diag_mask = torch.diag(torch.ones(epsilon_precision.shape[0]))
            epsilon_precision = diag_mask * torch.exp(torch.diag(epsilon_precision)) + (1. - diag_mask) * epsilon_precision
        self.epsilon_precision = nn.Parameter(epsilon_precision,requires_grad= True)

        # Linear Dynamic system of z_t = Az_{t-1} + epsilon_t
        self.LDS_transformation = nn.Linear(zt_dim,zt_dim,bias=False)
        if LDS_matrix is None:
            self.LDS_transformation.weight = nn.Parameter(torch.rand((zt_dim,zt_dim)),requires_grad=True)

        # Loading Transformation for poisson rate:
        self.loading_transformation  = nn.Linear(zt_dim,xt_dim)
        if loading_matrix is None:
            self.loading_transformation.weight = nn.Parameter(torch.rand(xt_dim,zt_dim),requires_grad=True)
        if loading_bias is None:
            self.loading_transformation.bias = nn.Parameter(torch.rand(xt_dim))

    def forward(self,x:torch.Tensor,z:torch.Tensor,batched =True):
        """
        This method must be
        :param x: a tensor of size dim_x vector corresponding to the observations of the time series
        :param z: a tensor of size batch_size T x dim_z corresponding to a sample drawn from the approximate posterior.
        :return: Expected Log Joint Likelihood: P_{\theta}(x,z) (averaged over the batches)
        """
        if batched:
            return torch.mean(
                torch.sum(MultivariateNormal(loc=self.LDS_transformation(z[:, :-1, :]),
                                             precision_matrix=self.epsilon_precision).log_prob(z[:, 1:, :]), dim=1) + \
                torch.sum(Poisson(rate=torch.exp(self.loading_transformation(z[:,1:,:]))).log_prob(x[1:,:]), dim=(1, 2)))
        else:
            return torch.mean(MultivariateNormal(loc = torch.zeros(z.shape[2]),precision_matrix=self.epsilon_precision).log_prob(z[:,0,:]) + \
            torch.sum(MultivariateNormal(loc = self.LDS_transformation(z[:,:-1,:]),
                                         precision_matrix= self.epsilon_precision).log_prob(z[:,1:,:]),dim = 1) + \
            torch.sum(Poisson(rate = torch.exp(self.loading_transformation(z))).log_prob(x),dim=(1,2)))
