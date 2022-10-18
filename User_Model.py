from torch import nn

class Joint_Likelihood(nn.Module):
    """
    Astract class that must be implemented by the user when specifying a model for the data.
    This class allows the user to specify the joint likelihood for a single sample z (for convenience)
    """
    def __init__(self, **named_parameters):
        """
        This method takes in a dictionary of model parameters
        :param named_model_parameters: keyword arguments
        """
        # registering the keyword arguments specified as parameters of the model
        super().__init__()
        for parameter_name in named_parameters:
            print(parameter_name)
            #print(parameter_name)
            #print(named_model_parameters[parameter_name])
            #self.vars()[parameter_name] = nn.Parameter(named_model_parameters[parameter_name],requires_grad= True)

    def forward(self,x,z):
        """
        This method must be
        :param x: a Tx dim_x vector corresponding to the observations of the time series
        :param z: a single sample of size T x dim_x corresponding to a sample drawn from the approximate posterior.
        :return: Joint-likelihood: P_{\theta}(x,z)
        """
        raise Exception("Implement me. This is an abstract method.")


class Expected_Joint_Likelihood(nn.Module):
    """
    Astract class that must be implemented by the user when specifying a model for the data.
    This class allows the user to directly implement
    """

    def __init__(self):
        """
        This method takes in a dictionary of model parameters
        :param model_parameters: varying length sequence of model parameters
        """
        # registering the keyword arguments specified as parameters of the model
        super().__init__()
        for parameter_name, in named_model_parameters:
            self.vars()[parameter_name] = nn.Parameter(named_model_parameters[parameter_name],requires_grad= True)

    def forward(self,x,z):
        """
        This method must be
        :param x: a Tx dim_x vector corresponding to the observations of the time series
        :param z: batch of samples of size batch x T x dim_x corresponding to a sample drawn from the approximate posterior.
        :return: Estimate of expected Joint-likelihood: E_{z~q}(P_{\theta}(x,z))
        """
        raise Exception("Implement me. This is an abstract method.")