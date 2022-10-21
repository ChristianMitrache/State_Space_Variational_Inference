import torch
from torch.nn import Module
from Approximate_Posterior import Approximate_Model
import Poisson_LDS
import User_Model
class Approximate_Inference_Loss(Module):

    def __init__(self, Approx_Model: Approximate_Model, User_Model: Module):
        super().__init__()
        self.Approx_Post = Approx_Model
        self.User_model = User_Model

    def forward(self,n,x):
        """
        Makes calls to approximate model class and estimates the KL divergence between the user model and
        and the approximate model.
        :param n: number of samples to take when estimating expectation
        :param x: observations of the time series (a tensor of shape batch_time x x_dim )
        :return: returns the KL divergence between approximate and user model.
        """
        samples = self.Approx_Post.Sample(n, x)  # Batch x time x x_dim
        return self.Approx_Post(x) + self.User_model(x,samples)




