import unittest

from Mean_Cov_Models import Inverse_Variance_Model
import torch
from Block_Functions import Compute_Block_Cholesky

class Test_Block_Cholesky(unittest.TestCase):
    def test_block_cholesky_diagonal(self):
        B = torch.zeros((9,1,1))
        D = torch.tensor([[[1.]], [[2.]], [[3.]], [[4.]], [[5.]], [[6.]], [[7.]], [[8.]], [[9.]], [[10.]]])
        chol_tuple = Compute_Block_Cholesky(D, B)
        self.assertTrue(torch.allclose(chol_tuple[0]**2, D) and torch.allclose(chol_tuple[1],B)) # add assertion here

    def test_block_cholesky_non_diagonal(self):
        D = torch.exp(torch.rand((10,1,1)))
        B = torch.rand((9,1,1))
        A = torch.diag(torch.squeeze(D)) + torch.diag_embed(torch.squeeze(B),offset  =-1)
        chol_tuple = Compute_Block_Cholesky(D, B)
        L_block_chol = torch.diag(torch.squeeze(chol_tuple[0])) + torch.diag_embed(torch.squeeze(chol_tuple[1]),offset  =-1)
        L_torch_chol = torch.linalg.cholesky_ex(A)[0]
        self.assertTrue(torch.allclose(L_torch_chol,L_block_chol)) # add assertion here

    def test_block_cholesky_non_diagonal_2(self):
        xt_dim = 5
        time = 1000
        L = torch.rand((time,xt_dim,xt_dim),dtype = torch.float64)+5
        D = torch.exp(L@ torch.transpose(L,dim0=1,dim1=2))
        B = torch.rand((time-1,xt_dim,xt_dim),dtype = torch.float64)+5
        # Constructing matrix to feed into torch from this:
        #cat_tensor = torch.cat((D[:-1, :, :], B), dim=1)
        # rearranging cat_tensor to represent it as 2-d Array
        chol_tuple = Compute_Block_Cholesky(D, B)
        A = reshape_helper_D(D, B,lower = False)
        L_block_chol = reshape_helper_D(chol_tuple[0],chol_tuple[1],lower = True)
        L_torch_chol = torch.linalg.cholesky_ex(A)[0]
        self.assertTrue(torch.allclose(L_torch_chol,L_block_chol)) # add assertion here

    def test_block_cholesky_non_diagonal_2_small_dim_stable(self):
        xt_dim = 3
        time = 5
        L = torch.rand((time,xt_dim,xt_dim),dtype = torch.float64)+5
        D = torch.exp(L@ torch.transpose(L,dim0=1,dim1=2))
        B = torch.rand((time-1,xt_dim,xt_dim),dtype = torch.float64)+5
        # Constructing matrix to feed into torch from this:
        #cat_tensor = torch.cat((D[:-1, :, :], B), dim=1)
        # rearranging cat_tensor to represent it as 2-d Array
        chol_tuple = Compute_Block_Cholesky(D, B)
        A = reshape_helper_D(D, B,lower = False)
        L_block_chol = reshape_helper_D(chol_tuple[0],chol_tuple[1],lower = True)
        L_torch_chol = torch.linalg.cholesky_ex(A)[0]
        self.assertTrue(torch.allclose(L_torch_chol,L_block_chol)) # add assertion here

    def test_block_cholesky_non_diagonal_2_small_dim_unstable(self):
        xt_dim = 3
        time = 5
        L = torch.rand((time,xt_dim,xt_dim),dtype = torch.float64)
        D = torch.exp(L@ torch.transpose(L,dim0=1,dim1=2))
        B = torch.rand((time-1,xt_dim,xt_dim),dtype = torch.float64)
        # Constructing matrix to feed into torch from this:
        #cat_tensor = torch.cat((D[:-1, :, :], B), dim=1)
        # rearranging cat_tensor to represent it as 2-d Array
        chol_tuple = Compute_Block_Cholesky(D, B)
        A = reshape_helper_D(D, B,lower = False)
        L_block_chol = reshape_helper_D(chol_tuple[0],chol_tuple[1],lower = True)
        L_torch_chol = torch.linalg.cholesky_ex(A)[0]
        print(torch.isclose(L_torch_chol,L_block_chol))
        self.assertTrue(torch.allclose(L_torch_chol,L_block_chol)) # add assertion here

def reshape_helper_D(D,B,lower = False):
    """
    helper function that reshapes tensor of block matrices to bigger matrix. This is for comparing my computed values
    against torch.
    :param D: Diagonal Block Tensor: Txnxn
    :param B: Below Diagonal Block Tensor (T-1)xnxn
    :return: (nT)x(nT) matrix containing diagonal and off diagonal blocks (which will also be symmetric)
    """
    xt_dim = D.shape[1]
    D_block_diag = torch.block_diag(*torch.split(D.reshape(D.shape[0] * D.shape[1], D.shape[2]), D.shape[1]))
    B_block_diag = torch.block_diag(*torch.split(B.reshape(B.shape[0] * B.shape[1], B.shape[2]), B.shape[1]))
    padding_func = torch.nn.ConstantPad2d((0,xt_dim, xt_dim, 0), 0)
    B_block_full = padding_func(B_block_diag)
    #B_block_full = torch.concat((torch.zeros(B_block_diag.shape[0] + xt_dim, xt_dim),
    #                             torch.concat((torch.zeros((xt_dim, B_block_diag.shape[0])),
    #                                           B_block_diag))), dim=1)
    if lower:
        return D_block_diag + B_block_full
    else:
        return D_block_diag + B_block_full + B_block_full.T
