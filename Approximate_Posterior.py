# Abstract Class format taken directly from:  https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
# with significant changes made.
import math

import torch
from torch import nn


class Recognition_Model(nn.Module):
    '''
    Recognition Model Interace Class
    Recognition model approximates the posterior given some observations
    Different forms of recognition models will have this interface

    '''

    def __init__(self):
        super().__init__()

    def eval_entropy(self,x):
        '''
        Evaluates entropy of posterior approximation
        H(q(x))
        This is NOT normalized by the number of samples
        '''
        raise Exception("Please implement me. This is an abstract method.")

    def get_Samples(self,n):
        '''
        Returns tensor of samples of size n x dim_z
        '''
        raise Exception("Please implement me. This is an abstract method.")


class Time_Series_Approx_Gaussian_First_Param(Recognition_Model):
    """
    This class of models takes in the log-likelihood and returns a differentiable estimate of L(theta,phi; X)
    """
    def __init__(self, time_steps,mean_model,cov_model):
        self.mean_model = mean_model
        self.cov_model = cov_model
        self.time_steps = time_steps
        super().__init__()

    def eval_entropy(self,x):
        '''
        Evaluating Gaussian entropy in equation (9) in paper and returning it. Given diagonal blocks of cholesky
        decomposition for the variance.
        '''
        # getting model outputs
        # getting cholesky factors for mean_cov_model
        D,B = self.cov_model(x)
        return (self.x_dim*self.time_steps/2)*(1 + torch.log(2*torch.tensor(math.pi))) +\
               torch.sum(torch.log(torch.diagonal(D,dim1 = 1,dim2=2)))

    def get_Sample(self,n,x):
        """
        Implementing function that samples from approximate posterior.
        :return: Returns tensor of samples of size n x T x dim_z
        """
        with torch.no_grad:
            mean = self.mean_model(x)
            epsilon = torch.normal(size=(n,self.time_steps,self.cov_model.xt_dim))
            noise_cov_term=  self.cov_model
            #noise_cov_term = self.cov_model.block_inverse_solve(self.D, self.B, epsilon)
            return torch.unsqueeze(mean,dim =0) + noise_cov_term

    def forward(self,x):
        """
        Implementing forward pass required by nn module. This gives the.
        :param x:
        :return: the entropy of the
        """
        self.D,self.B = self.cov_model(x)
        return (self.x_dim*self.time_steps/2)*(1 + torch.log(2*torch.tensor(math.pi))) +\
               torch.sum(torch.log(torch.diagonal(self.D,dim1 = 1,dim2=2)))