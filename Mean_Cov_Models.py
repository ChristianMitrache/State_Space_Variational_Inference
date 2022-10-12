import torch
import torch.nn as nn
import functorch

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

class Variance_Model(nn.Module):
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
        matrix.
        """

        x = torch.unsqueeze(x, dim=2)
        B, L = self.model_B(self.initial_weights_B * x), self.model_L(self.initial_weights_L* x)
        # Making sure D is strictly positive definite,symmetric
        # Not sure if it's faster do the unsqueezing below or to just deal with this using masking - will have to test this
        L = torch.tril(L,diagonal=-1) + (torch.unsqueeze(torch.exp(torch.diagonal(L,dim1 = 1,dim2=2)),dim = 2)*torch.eye(L.shape[1]))
        D = L@torch.transpose(L,dim0=1,dim1=2)
        return self.compute_block_cholesky(D,B)

    def compute_block_cholesky(self ,D,B):
        """
        This uses recursive relation given from:
        https://en.wikipedia.org/wiki/Block_LU_decomposition#:~:text=Block%20Cholesky%20decomposition&text=is%20a%20matrix%20whose%20elements%20are%20all%20zero.

        :param D: Txnxn tensor representing T nxn diagonal blocks of covariance matrix.
        :param B: (T-1)xnxn tensor representing B nxn lower diagonal blocks of covariance matrix.
        :return: (diag,off_diag) where dim(diag): Txnxn and dim(off_diag): (T-1)xnxn
        """
        diag_chol_blocks = []
        off_diag_chol_blocks =[]
        #Initial computation for
        A = D[0, :, :] # Compute left-upper block (the very first one)
        for i in range(0,D.shape[0]-1):
            L_A =torch.linalg.cholesky_ex(A)[0]
            diag_chol_blocks.append(L_A)
            B_i = B[i, :, :]
            bottom_left = torch.linalg.solve_triangular(L_A.T,B_i,left = False,upper= True)
            off_diag_chol_blocks.append(bottom_left)
            A = D[i+1,:,:] - bottom_left @ bottom_left.T # Compute left-lower block of cholesky block
        diag_chol_blocks.append(torch.linalg.cholesky_ex(A)[0]) # appending last diagonal chol block
        diag_chol_blocks = torch.stack(diag_chol_blocks,dim = 0)
        off_diag_chol_blocks= torch.stack(off_diag_chol_blocks,dim= 0)

        return (diag_chol_blocks,off_diag_chol_blocks)

    def block_inverse_solve(self,D,B,x):
        """
        Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, solve for L^{-1}x where L is cholesky
        decomposition
        :param D: cholesky diagonal blocks tensor with shape Txnxn
        :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
        :param x: tensor with shape mxTxn where m is the batch dimension.
        :return: mxTxn tensor which solves the m is batch dimension.
        """

        #TODO

    def block_mat_vec(self,D,B,x):
        """
        Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, compute matrix vector product
        :param D: cholesky diagonal blocks tensor with shape Txnxn
        :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
        :param x: tensor with shape mxTxn where m is the batch dimension.
        :return: mxTxn tensor (m is batch dimension).
        """

        #TODO