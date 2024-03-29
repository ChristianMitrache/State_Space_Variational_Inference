import torch
from functorch import vmap
from typing import Tuple
import numpy as np
# This file contains important functions for computing block matrix operations without storing the entire matrix.

@torch.jit.script
def Compute_Block_Cholesky(D: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    This uses recursive relation given from:
    https://en.wikipedia.org/wiki/Block_LU_decomposition#:~:text=Block%20Cholesky%20decomposition&text=is%20a%20matrix%20whose%20elements%20are%20all%20zero.

    :param D: Txnxn tensor representing T nxn diagonal blocks of covariance matrix.
    :param B: (T-1)xnxn tensor representing B nxn lower diagonal blocks of covariance matrix.
    :return: (diag,off_diag) where dim(diag): Txnxn and dim(off_diag): (T-1)xnxn
    """
    diag_chol_blocks = torch.empty(D.shape,dtype=D.dtype)
    off_diag_chol_blocks = torch.empty(B.shape,dtype = B.dtype)
    # Initial computation for
    A = D[0, :, :]
    for i in range(0, D.shape[0] - 1):
        L_A = torch.linalg.cholesky_ex(A)[0]
        diag_chol_blocks[i, :, :] = L_A
        bottom_left = torch.linalg.solve_triangular(L_A.T, B[i, :, :], left=False, upper=True)
        off_diag_chol_blocks[i,:,:] = bottom_left
        A = D[i + 1, :, :] - bottom_left @ bottom_left.T  # Compute left-lower block of cholesky block

    diag_chol_blocks[D.shape[0]-1,:,:]  = torch.linalg.cholesky_ex(A)[0]  # appending last diagonal chol block
    return (diag_chol_blocks, off_diag_chol_blocks)

@torch.jit.script
def Block_Triangular_Solve_new(D:torch.Tensor, B:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    """
    Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, solve for L^{-1}x where L is cholesky
    lower triangle.

    Note: THIS FUNCTION MUTATES THE D matrix!
    :param D: cholesky diagonal blocks tensor with shape Txnxn
    :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor which solves the m is batch dimension.
    """
    diag_index = torch.arange(start=0, end=D.shape[1])
    diagonals = D[:, diag_index, diag_index]
    D[:, diag_index, diag_index] = 1
    return_vec = torch.empty((x.shape[0],x.shape[1],x.shape[2]),dtype = x.dtype)
    # Initial solution (without block B_i)
    solve_i = torch.linalg.solve_triangular(D[0,:,:], x[:,0,:].T,upper = False,unitriangular = True).T
    return_vec[:,0,:] = solve_i
    # Iterating through rest of blocks with B_i, D_i present in linear system
    for i in range(1,D.shape[0]):
        solve_i = torch.linalg.solve_triangular(D[i,:,:], x[:,i,:].T- B[i-1,:,:]@ solve_i.T,upper = False, unitriangular = True).T
        return_vec[:,i,:] = solve_i
    return return_vec/diagonals

@torch.jit.script
def Block_Triangular_Solve(D:torch.Tensor, B:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    """
    Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, solve for L^{-1}x where L is cholesky
    lower triangle.
    :param D: cholesky diagonal blocks tensor with shape Txnxn
    :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor which solves the m is batch dimension.
    """
    return_vec = torch.empty((x.shape[0],x.shape[1],x.shape[2]),dtype = x.dtype)
    # Initial solution (without block B_i)
    solve_i = torch.linalg.solve_triangular(D[0,:,:], x[:,0,:].T,upper = False).T
    return_vec[:,0,:] = solve_i
    # Iterating through rest of blocks with B_i, D_i present in linear system
    for i in range(1,D.shape[0]):
        solve_i = torch.linalg.solve_triangular(D[i,:,:], x[:,i,:].T- B[i-1,:,:]@ solve_i.T,upper = False).T
        return_vec[:,i,:] = solve_i
    return return_vec

@torch.jit.script
def Block_Lower_Triangle(x:torch.Tensor,dim: int) -> torch.Tensor:
    """
        Function that takes in a vector of size T*n*(n+1)/2 and outputs a corresponding batch of
    lower triangles.

    :param x: tensor of size (T) x (n*(n+1)/2)
    :param T: batch size (in time)
    :param dim: dimension of latent vector
    :return: returns a tensor of size (T x dim x dim) representing the batches of the entries of the lower triangles.
    """
    return_tensor = torch.zeros(x.shape[0],dim,dim)
    torch_indices = torch.tril_indices(dim, dim, offset=0)
    return_tensor[:,torch_indices[0],torch_indices[1]] = x
    return return_tensor



def Block_Lower_Triangle_Mat_Vec(L_D:torch.Tensor, L_B:torch.Tensor, x:torch.Tensor)-> torch.Tensor:
    """
    Given cholesky diagonal blocks L_D, cholesky off-diagonal blocks L_B, compute matrix vector product in batches
    :param L_D: cholesky diagonal blocks tensor with shape Txnxn
    :param L_B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor (m is batch dimension).
    """
    batched_mat_vec_mul = vmap(torch.mv)

    def batched_fix_L_D(x: torch.Tensor):
        """
        fixes matrix D so I can batch over x now.
        :param x: tensor that is Txn
        :return: returns batched matrix vector product of Dx
        """
        return batched_mat_vec_mul(L_D, x)

    def batched_fix_L_B(x:torch.Tensor):
        """
        fixes matrix B so I can batch over x now.
        :param x: tensor that is (T-1)xn
        :return: returns batched matrix vector product of Bx
        """
        return batched_mat_vec_mul(L_B, x)


    mat_vec_multiply_D = vmap(batched_fix_L_D)
    mat_vec_multiply_B = vmap(batched_fix_L_B)
    B_lower_sum = torch.cat((torch.zeros((x.shape[0], 1, x.shape[2])), mat_vec_multiply_B(x[:, 1:, :])), dim=1)
    return mat_vec_multiply_D(x) + B_lower_sum