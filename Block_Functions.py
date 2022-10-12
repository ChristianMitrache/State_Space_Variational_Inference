import torch
# import functorch <- I don't need this yet

# This file contains important functions for computing block matrix operations without storing the entire matrix.

def compute_block_cholesky(D, B):
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


def block_inverse_solve(D, B, x):
    """
    Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, solve for L^{-1}x where L is cholesky
    decomposition
    :param D: cholesky diagonal blocks tensor with shape Txnxn
    :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor which solves the m is batch dimension.
    """

    # TODO


def block_mat_vec(self, D, B, x):
    """
    Given cholesky diagonal blocks D, cholesky off-diagonal blocks B, compute matrix vector product
    :param D: cholesky diagonal blocks tensor with shape Txnxn
    :param B: cholesky off-diagonal blocks with shape (T-1)xnxn
    :param x: tensor with shape mxTxn where m is the batch dimension.
    :return: mxTxn tensor (m is batch dimension).
    """

    # TODO