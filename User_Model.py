import torch
from torch import nn
from typing import Dict

# NOT SURE IF I AM GONNA KEEP THIS CLASS - KINDA SEEMS LIKE A WASTE OF TIME.

@torch.jit.script
class Expected_Joint_Likelihood(nn.Module):
    """
    Astract class that must be implemented by the user when specifying a model for the data.
    This classes only function is to remind the user what needs to be implemented in the forward pass and
    to force the user to register the model parameters.
    """
    def __init__(self, named_model_parameters: Dict[str,torch.Tensor(float)]):
        """
        This method takes in a dictionary of model parameters
        :param named_model_parameters: keyword arguments
        """
        # registering the keyword arguments specified as parameters of the model
        super().__init__()
        self.params = nn.ParameterDict(named_model_parameters)

    def forward(self,x,z) -> torch.Tensor:
        """
        This method must be
        :param x: a Tx dim_x vector corresponding to the observations of the time series
        :param z: a single sample of size T x dim_x corresponding to a sample drawn from the approximate posterior.
        :return: Joint-likelihood: P_{\theta}(x,z)
        """
        raise Exception("Implement me. This is an abstract method.")
