# Abstract Class format taken directly from:  https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
# with significant changes made.
import math
from Block_Functions import Block_Triangular_Solve
import torch
from torch import nn


class Approximate_Model(nn.Module):
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

    def Sample(self, n, x):
        '''
        Returns tensor of samples of size n x dim_z
        '''
        raise Exception("Please implement me. This is an abstract method.")


class Time_Series_Approx_Gaussian_First_Param(Approximate_Model):
    """
    This class implements the time series approximate Gaussian parameterization for variational inference.

    This class stores class variables:
    :param mean_model: model for parameterizing the mean of gaussian at time t.
    :param inv_cov_model: model that produces lower triangular matrix of cholesky decomposition of the inverse variance.
           (It is required that it returns this in the form of B,D)

    AFTER CALLING FORWARD PASS TO COMPUTE ENTROPY:
    :param L_D: diagonal blocks of the cholesky decomposition of the inverse covariance matrix
    :param L_B off-diagonal blocks of the cholesky decomposition of the inverse covariance matrix
    """
    def __init__(self,mean_model,inv_cov_model):
        super().__init__()
        self.mean_model = mean_model
        self.inv_cov_model = inv_cov_model

    def Sample(self, n, x):
        """
        Returns batched samples from time series model given observational vector x.
        Note: DO NOT CALL THIS METHOD BEFORE forward for this function.
        :param n: number of batches.
        :param x: observational time series x
        :return: Returns n x T x x_dim tensor of normal samples
        """
        mean = self.mean_model(x)
        epsilon = torch.normal(mean= 0, std= 1,size=(n, mean.shape[0], mean.shape[1]),requires_grad= False)
        noise_cov_term = Block_Triangular_Solve(self.L_D, self.L_B, epsilon)
        return torch.unsqueeze(mean,dim = 0) + noise_cov_term

    def forward(self,x):
        """
        Implementing forward pass required by nn module.
        :param x: observations from time series (Should be a time x dim_x vector)
        :return:  entropy value of the normal distributions centered at each data-point in full time series.
        """
        self.L_D,self.L_B = self.inv_cov_model(x)
        return (x.shape[0]*x.shape[1]/2)*(1 + torch.log(2*torch.tensor(math.pi))) - \
               torch.sum(torch.log(torch.diagonal(self.L_D,dim1 = 1,dim2=2)),dim=1)

    def get_mean_parameters(self):
        """
        This function returns the mean parameters used by the mean NN.
        :return: returns parameters from torch.parameters call
        """
        return self.mean_model.parameters(recurse = True)

    def get_variance_parameters(self):
        """
        This function returns the variance parameters used by the variance NN.
        :return: returns parameters from torch.parameters call
        """
        return self.inv_cov_model.parameters(recurse = True)