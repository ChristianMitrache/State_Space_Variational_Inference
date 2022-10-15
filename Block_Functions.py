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
    diag_chol_blocks = []
    off_diag_chol_blocks = []
    # Initial computation for
    A = D[0, :, :]  # Compute left-upper block (the very first one)
    for i in range(0, D.shape[0] - 1):
        L_A = torch.linalg.cholesky_ex(A)[0]
        diag_chol_blocks.append(L_A)
        B_i = B[i, :, :]
        bottom_left = torch.linalg.solve_triangular(L_A.T, B_i, left=False, upper=True)
        off_diag_chol_blocks.append(bottom_left)
        A = D[i + 1, :, :] - bottom_left @ bottom_left.T  # Compute left-lower block of cholesky block
    diag_chol_blocks.append(torch.linalg.cholesky_ex(A)[0])  # appending last diagonal chol block
    diag_chol_blocks = torch.stack(diag_chol_blocks, dim=0)
    off_diag_chol_blocks = torch.stack(off_diag_chol_blocks, dim=0)

    return (diag_chol_blocks, off_diag_chol_blocks)


def Block_Inverse_Solve(D, B, x):
    """
    Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, solve for L^{-1}x where L is cholesky
    decomposition
    :param D: cholesky diagonal blocks tensor with shape Txnxn
    :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor which solves the m is batch dimension.
    """

    # TODO


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


    #D = torch.unsqueeze(D,0)
    #B = torch.unsqueeze(B,0)
    #x = torch.unsqueeze(x,3)
    #return torch.sum(D*x,dim = 3) + \
    #       torch.cat((torch.zeros((x.shape[0],1,x.shape[2])),torch.sum(B*x[:,1:,:,:],dim = 3)),dim = 1)

