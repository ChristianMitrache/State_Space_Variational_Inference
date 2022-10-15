# Abstract Class format taken directly from:  https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
# with significant changes made.
import math
from Block_Functions import Block_Mat_Vec
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

    def Sample(self,n,x):
        '''
        Returns tensor of samples of size n x dim_z
        '''
        raise Exception("Please implement me. This is an abstract method.")


class Time_Series_Approx_Gaussian_First_Param(Recognition_Model):
    """
    This class of models takes in the log-likelihood and returns a differentiable estimate of L(theta,phi; X)
    """
    def __init__(self,mean_model,inv_cov_model):
        super().__init__()
        self.mean_model = mean_model
        self.inv_cov_model = inv_cov_model

    def Sample(self,n,x):
        """
        Returns batched samples from time series model given observational vector x.
        Note: DO NOT CALL THIS METHOD BEFORE forward for this function.
        :param n: number of batches.
        :param x: observational time series x
        :return: Returns n x T x x_dim tensor of normal samples
        """
        epsilon = torch.normal(mean= 0, std= 1,size=(n, x.shape[0], x.shape[1]),requires_grad= False)
        noise_cov_term = Block_Mat_Vec(self.D, self.B, epsilon)
        mean = self.mean_model(x)
        return torch.unsqueeze(mean,dim = 0) + noise_cov_term

    def forward(self,x):
        """
        Implementing forward pass required by nn module.
        :param x: observations from time series (Should be a T x dim_x vector)
        :return:  entropy value of the normal distributions centered at each data-point in time series.
        """
        self.D,self.B = self.inv_cov_model(x)
        return (x.shape[0]*x.shape[1]/2)*(1 + torch.log(2*torch.tensor(math.pi))) - \
               torch.sum(torch.log(torch.diagonal(self.D,dim1 = 1,dim2=2)),dim=1)