import torch
import torch.nn as nn
from Block_Functions import Compute_Block_Cholesky

class Fully_Connected_Model(nn.Module):
    '''
    This module implements simple dense network with linear layer dimensions and non-linearity of user's choice.

    Note: For forward pass, input should be of size T X N (batch x dimension)
    '''
    def __init__(self,xt_dim,linear_layer_dims,non_lin_module):
        super().__init__()
        # creating sequential module which will be called later:
        # Note: Batch norms are not included might be interesting to experiment with this later
        self.linear_layers= nn.Sequential(nn.Linear(xt_dim,linear_layer_dims[0],bias= False),
                                          non_lin_module)
        #nn.BatchNorm1d(linear_layer_dims[0])
        for i in range(len(linear_layer_dims)-1):
            self.linear_layers.append(nn.Linear(linear_layer_dims[i],linear_layer_dims[i+1],bias=False))
            self.linear_layers.append(non_lin_module)
            #self.linear_layers.append(nn.BatchNorm1d(linear_layer_dims[i+1]))
        # appending last layer which maps back to xt_dim
        self.linear_layers.append(nn.Linear(linear_layer_dims[-1],xt_dim,bias= False))
        self.linear_layers.append(non_lin_module)
        #self.linear_layers.append(nn.BatchNorm1d(xt_dim))


    def forward(self, x):
        return self.linear_layers(x)
"""
For now Don't worry about this
class Fully_Connected_PD_Model(Fully_Connected_Model):
"""
    #This Model Implements a model from A -> A' where A,A' are positive Definite Matrices.
    #To enforce this constraint the linear layers are multiplied with each other.

"""
    def __init__(self,xt_dim,linear_layer_dims,non_lin_module):
        super().__init__(xt_dim,linear_layer_dims,non_lin_module)

    def forward(self, x):
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            x = x@ x.T + torch.eye(x.shape[0])*(1E-15) # Enforcing postive definite constraint for each layer
        return x
"""

class Fully_Connected_Mean_Model(Fully_Connected_Model):
    '''
    This module implements the mapping x_t -> mu_{phi}(x_t).
    '''
    def __init__(self,xt_dim,linear_layer_dims,non_lin_module):
        super().__init__(xt_dim,linear_layer_dims,non_lin_module)

class Inverse_Variance_Model(nn.Module):
    '''
    Variance model for parameterization 1 of the variational inference for state space models paper.
    models the block diagonal and off-block diagonal matrices for the lower-triangle of the cholesky decomposition
    of the covariance.

    Note: As defined below, the PARAMETERS of network are not functions of time.
    '''

    def __init__(self,xt_dim,linear_layer_dims,non_lin_module):
        super().__init__()
        self.xt_dim = xt_dim
        self.model_B = Fully_Connected_Model(xt_dim,linear_layer_dims,non_lin_module)
        self.initial_weights_B = nn.Parameter(torch.rand(size=(xt_dim, xt_dim), requires_grad=True))
        self.model_L = Fully_Connected_Model(xt_dim, linear_layer_dims, non_lin_module)
        self.initial_weights_L = nn.Parameter(torch.rand(size=(xt_dim, xt_dim), requires_grad=True))

    def forward(self, x):
        """
        Computes forward pass and returns diagonal blocks of block cholesky decomp.
        x should be a Txn tensor.
        returns a tuple of ((Tx n x n), (T-1 x n x n)) matrices representing the block matrices for the lower diagonal
        matrix of the inverse covariance.
        """

        x = torch.unsqueeze(x, dim=2)
        B, L = self.model_B(self.initial_weights_B * x), self.model_L(self.initial_weights_L* x)
        # Making sure D is strictly positive definite,symmetric
        # Not sure if it's faster do the unsqueezing below or to just deal with this using masking - will have to test this
        L = torch.tril(L,diagonal=-1) + (torch.unsqueeze(torch.exp(torch.diagonal(L,dim1 = 1,dim2=2)),dim = 2)*torch.eye(L.shape[1]))
        D = L@torch.transpose(L,dim0=1,dim1=2)
        return Compute_Block_Cholesky(D, B)