import torch
import torch.nn as nn
from Block_Functions import Compute_Block_Cholesky

class Fully_Connected_Model(nn.Module):
    '''
    This module implements simple dense network with linear layer dimensions and non-linearity of user's choice.
    '''
    def __init__(self,xt_dim,zt_dim,linear_layer_dims,non_lin_module):
        super().__init__()
        # creating sequential module which will be called later:
        # Note: Batch norms are not included might be interesting to experiment with this later
        self.linear_layers= nn.Sequential(nn.Linear(xt_dim,linear_layer_dims[0],bias= False),
                                          non_lin_module)
        #nn.BatchNorm1d(linear_layer_dims[0])
        for i in range(len(linear_layer_dims)-1):
            self.linear_layers.append(nn.Linear(linear_layer_dims[i],linear_layer_dims[i+1],bias=False))
            self.linear_layers.append(non_lin_module)
            #self.linear_layers.append(nn.BatchNorm1d(linear_layer_dims[i+1])) <- not sure if this is desired
        # appending last layer which maps back to xt_dim
        self.linear_layers.append(nn.Linear(linear_layer_dims[-1],zt_dim,bias= False))
        self.linear_layers.append(non_lin_module)
        #self.linear_layers.append(nn.BatchNorm1d(xt_dim)) <- not sure if this is desired


    def forward(self, x):
        """
        Returns
        :param x: tensor of size T x x_dim
        :return: tensor of size T x z_dim
        """
        return self.linear_layers(x)

class Fully_Connected_Mean_Model(Fully_Connected_Model):
    '''
    This module implements the mapping x_t -> mu_{phi}(x_t).
    '''
    def __init__(self,xt_dim,zt_dim,linear_layer_dims,non_lin_module):
        super().__init__(xt_dim,zt_dim,linear_layer_dims,non_lin_module)

class Inverse_Variance_Model(nn.Module):
    '''
    Variance model for parameterization 1 of the variational inference for state space models paper.
    models the block diagonal and off-block diagonal matrices for the lower-triangle of the cholesky decomposition
    of the covariance.

    Note: As defined below, the PARAMETERS of network are not functions of time.
    '''

    def __init__(self,xt_dim,zt_dim,linear_layer_dims_B,linear_layer_dims_D,non_lin_module):
        """
        This function iniatilizes Inverse Variance model.
        :param zt_dim: dimension of z_t
        :param linear_layer_dims_B: dimensions of linear layers for network producing the B_t:
        NOTE: the values passed here should be gradually changing from x_t_dim --> zt_dim**2
        :param linear_layer_dims_D: dimensions of linear layers for network producing the D_t:
        NOTE: the values passed here should be gradually changing from x_t_dim --> zt_dim**2
        :param non_lin_module: the non-linearities to be used in both networks. (must be initialized nn.module)
        """
        super().__init__()
        self.zt_dim = zt_dim
        self.model_B = Fully_Connected_Model(xt_dim,(zt_dim)**2,linear_layer_dims_B,non_lin_module)
        self.final_weights_B_z = nn.Parameter(torch.rand(size=(zt_dim, zt_dim), requires_grad=True))
        self.model_D = Fully_Connected_Model(xt_dim,zt_dim**2, linear_layer_dims_D, non_lin_module)
        self.final_weights_D_z = nn.Parameter(torch.rand(size=(zt_dim, zt_dim), requires_grad=True))
        self.non_linearity = non_lin_module
        self.batch_norm = torch.nn.BatchNorm1d(zt_dim,track_running_stats= False)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        """
        Computes forward pass and returns diagonal blocks of block cholesky decomp.
        x should be a Txn tensor.

        NOTE: THIS FUNCTION MIGHT NOT WORK FOR BATCHED TIME SERIES.

        returns a tuple of ((Tx n x n), (T-1 x n x n)) matrices representing the block matrices for the lower diagonal
        matrix of the inverse covariance.
        """

        # reshaping before passing into linear models (not sure if I should do the reshaping after - maybe incorporate
        # some 2d convolutions? -consider convolutional network with large convolutional range?)

        # Designing architecture to produce numerically stable matrices for cholesky decomposition. (by applying a batch normalization after)
        B = self.final_weights_B_z * torch.reshape(self.model_B(x[:-1,:]),(x.shape[0]-1,self.zt_dim,self.zt_dim))
        D  = self.final_weights_D_z * torch.reshape(self.model_D(x), (x.shape[0],self.zt_dim, self.zt_dim))

        # Making sure D is strictly positive definite,symmetric
        # Not sure if it's faster do the unsqueezing below or to just deal with this using masking - will have to test this
        D = (10/D.shape[1])*(D @ torch.transpose(D, dim0=1, dim1=2)) # Making Matrix symmetric and normalizing values closer to N(1,1)

        D = torch.tril(D,diagonal=-1) + (torch.unsqueeze(torch.abs(torch.diagonal(D,dim1 = 1,dim2=2)),dim = 2)*torch.eye(D.shape[1])) + \
            torch.transpose(torch.tril(D,diagonal=-1),dim0=1,dim1=2)

        #D = torch.exp(-D)
        return Compute_Block_Cholesky(D, B)