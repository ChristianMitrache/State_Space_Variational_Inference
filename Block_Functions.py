import torch
from functorch import vmap

# This file contains important functions for computing block matrix operations without storing the entire matrix.

def Compute_Block_Cholesky(D, B):
    """
    This uses recursive relation given from:
    https://en.wikipedia.org/wiki/Block_LU_decomposition#:~:text=Block%20Cholesky%20decomposition&text=is%20a%20matrix%20whose%20elements%20are%20all%20zero.

    :param D: Txnxn tensor representing T nxn diagonal blocks of covariance matrix.
    :param B: (T-1)xnxn tensor representing B nxn lower diagonal blocks of covariance matrix.
    :return: (diag,off_diag) where dim(diag): Txnxn and dim(off_diag): (T-1)xnxn
    """
    diag_chol_blocks = torch.zeros(D.shape)
    off_diag_chol_blocks = torch.zeros(B.shape)
    # Initial computation for
    A = D[0, :, :]  # Compute left-upper block (the very first one)
    for i in range(0, D.shape[0] - 1):
        L_A = torch.linalg.cholesky_ex(A)[0]
        diag_chol_blocks[i,:,:] = L_A
        B_i = B[i, :, :]
        bottom_left = torch.linalg.solve_triangular(L_A.T, B_i, left=False, upper=True)
        off_diag_chol_blocks[i,:,:] = bottom_left
        A = D[i + 1, :, :] - bottom_left @ bottom_left.T  # Compute left-lower block of cholesky block

    diag_chol_blocks[D.shape[0],:,:]  = torch.linalg.cholesky_ex(A)[0]  # appending last diagonal chol block

    #diag_chol_blocks = torch.stack(diag_chol_blocks, dim=0)
    #off_diag_chol_blocks = torch.stack(off_diag_chol_blocks, dim=0)

    return (diag_chol_blocks, off_diag_chol_blocks)

def Block_Inverse_Solve(D, B, x):
    """
    Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, solve for L^{-1}x where L is cholesky
    decomposition lower triangle
    :param D: cholesky diagonal blocks tensor with shape Txnxn
    :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor which solves the m is batch dimension.
    """
    return_vec = torch.zeros((x.shape[0],x.shape[1],x.shape[2]))
    x = torch.unsqueeze(x,dim = 2)
    # Initial solution (without block B_i)
    x_i = x[:,0,:,:]
    solve_i = torch.linalg.solve_triangular(D[0,:,:],x_i,upper = False,left = False)
    return_vec[:,0,:] = solve_i
    # Iterating through rest of blocks with B_i, D_i present in linear system
    for i in range(1,D.shape[0]):
        x_i = x[:, i, :, :]
        solve_i = torch.linalg.solve_triangular(D[i,:,:],x_i-solve_i @ B[i-1,:,:],upper = False,left = False)
        return_vec[:,i,:] = solve_i
    return return_vec



def Block_Mat_Vec(D, B, x):
    """
    Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, compute matrix vector product in batches
    :param D: cholesky diagonal blocks tensor with shape Txnxn
    :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor (m is batch dimension).
    """
    batched_mat_vec_mul = vmap(torch.mv)

    def batched_fix_D(x):
        """
        fixes matrix D so I can batch over x now.
        :param x: tensor that is Txn
        :return: returns batched matrix vector product of Dx
        """
        return batched_mat_vec_mul(D,x)

    def batched_fix_B(x):
        """
        fixes matrix B so I can batch over x now.
        :param x: tensor that is (T-1)xn
        :return: returns batched matrix vector product of Bx
        """
        return batched_mat_vec_mul(B,x)

    mat_vec_multiply_D = vmap(batched_fix_D)
    mat_vec_multiply_B = vmap(batched_fix_B)
    return mat_vec_multiply_D(x) + \
           torch.cat((torch.zeros((x.shape[0],1,x.shape[2])),mat_vec_multiply_B(x[:,1:,:])),dim =1)

