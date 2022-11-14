import unittest
from timeit import timeit
from Mean_Cov_Models import Inverse_Variance_Model
import torch
from Block_Functions import Compute_Block_Cholesky,Block_Mat_Vec_Cholesky,Block_Triangular_Solve


def reshape_helper_D(L_D, L_B, lower = True):
    """
    helper function that reshapes tensor of block matrices to bigger matrix. This is for comparing my computed values
    against torch. REMEMBER: D,B ARE CHOLESKY DECOMPOSITION COMPONENTS.
    :param L_D: DIAGONAL Block Tensor: Txnxn
    :param L_B: Below Diagonal Block Tensor (T-1)xnxn
    :return: (nT)x(nT) matrix containing diagonal and off diagonal blocks (which will also be symmetric)
    """
    xt_dim = L_D.shape[1]
    D_block_diag = torch.block_diag(*torch.split(L_D.reshape(L_D.shape[0] * L_D.shape[1], L_D.shape[2]), L_D.shape[1]))
    B_block_diag = torch.block_diag(*torch.split(L_B.reshape(L_B.shape[0] * L_B.shape[1], L_B.shape[2]), L_B.shape[1]))
    padding_func = torch.nn.ConstantPad2d((0,xt_dim, xt_dim, 0), 0)
    B_block_full = padding_func(B_block_diag)
    #B_block_full = torch.concat((torch.zeros(B_block_diag.shape[0] + xt_dim, xt_dim),
    #                             torch.concat((torch.zeros((xt_dim, B_block_diag.shape[0])),
    #                                           B_block_diag))), dim=1)
    if lower:
        return D_block_diag + B_block_full
    else:
        return D_block_diag + B_block_full + B_block_full.T

######################################################################################################################

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
        L = torch.rand((time,xt_dim,xt_dim),dtype = torch.float64)+2
        D = torch.exp(L@ torch.transpose(L,dim0=1,dim1=2))
        B = torch.rand((time-1,xt_dim,xt_dim),dtype = torch.float64)+2
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
        # rearranging cat_tensor to represent it as 2-d Array
        chol_tuple = Compute_Block_Cholesky(D, B)
        A = reshape_helper_D(D, B,lower = False)
        L_block_chol = reshape_helper_D(chol_tuple[0],chol_tuple[1],lower = True)
        L_torch_chol = torch.linalg.cholesky_ex(A)[0]
        self.assertTrue(torch.allclose(L_torch_chol,L_block_chol)) # add assertion here

    def test_block_cholesky_non_diagonal_2_small_dim_unstable(self):
        xt_dim = 5
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

#############################################################################################################

class Test_Block_Mat_Vec_Cholesky(unittest.TestCase):
    """
    Testing the block matrix vector multiplication with values given by torch.
    """
    def test_block_Mat_Vec_Cholesky_small(self):
        # The reshaping into the big matrix for torch's matrix multiplication might cause a memory problem lol
        xt_dim = 5
        time = 10
        batch = 1
        L = torch.rand((time,xt_dim,xt_dim))+5
        D = torch.exp(L@ torch.transpose(L,dim0=1,dim1=2))
        #B = torch.zeros(((time-1,xt_dim,xt_dim)))
        B = torch.rand((time-1,xt_dim,xt_dim))+5
        full_vec = torch.rand((batch, time, xt_dim))
        mat_vec = Block_Mat_Vec_Cholesky(D, B, full_vec)
        full_mat = reshape_helper_D(D, B,lower = False)
        full_vec_squeeze = torch.reshape(full_vec,(batch,time*xt_dim))
        full_mat_vec = full_mat@full_vec_squeeze.T
        mat_vec = torch.reshape(mat_vec,(batch,xt_dim*time))
        self.assertTrue(torch.allclose(full_mat_vec.T,mat_vec)) # add assertion here


    def test_block_Mat_Vec_Cholesky_big(self):
        # The reshaping into the big matrix for torch's matrix multiplication might cause a memory problem lol
        xt_dim = 100
        time = 100
        batch = 30
        L = torch.rand((time,xt_dim,xt_dim))+5
        D = torch.exp(L@ torch.transpose(L,dim0=1,dim1=2))
        #B = torch.zeros(((time-1,xt_dim,xt_dim)))
        B = torch.rand((time-1,xt_dim,xt_dim))+5
        full_vec = torch.rand((batch, time, xt_dim))
        mat_vec = Block_Mat_Vec_Cholesky(D, B, full_vec)
        full_mat = reshape_helper_D(D, B,lower = False)
        full_vec_squeeze = torch.reshape(full_vec,(batch,time*xt_dim))
        full_mat_vec = full_mat@full_vec_squeeze.T
        mat_vec = torch.reshape(mat_vec,(batch,xt_dim*time))
        self.assertTrue(torch.allclose(full_mat_vec.T,mat_vec)) # add assertion here

#############################################################################################################

class Test_Block_Inverse_Solve(unittest.TestCase):
    """
    Testing the block matrix vector multiplication with values given by torch.
    """

    def test_Block_Triangular_Solve_small(self):
        """
        tests the block triangular solve and compares it to fully dense inverse solve implement in torch.
        """
        xt_dim = 3
        time = 4
        batch = 2
        L = torch.rand((time,xt_dim,xt_dim))
        D = torch.exp(L@ torch.transpose(L,dim0=1,dim1=2))+5
        B = torch.rand((time-1,xt_dim,xt_dim))

        # Need to construct cholesky decomposition to get diagonal matrix that is lower-triangular
        D,B = Compute_Block_Cholesky(D,B)
        full_vec = torch.rand((batch, time, xt_dim))
        # Computing block inverse
        Block_Solve = Block_Triangular_Solve(D, B, full_vec)
        # Reshaping to compute inverse with torch
        full_mat = reshape_helper_D(D, B,lower = True)
        full_vec_squeeze = torch.reshape(full_vec,(batch,time*xt_dim))
        # Computing Inverse with torch
        torch_inverse_solve = torch.linalg.solve(torch.repeat_interleave(torch.unsqueeze(full_mat,0)
                                                                         ,repeats = batch,dim =0), full_vec_squeeze)
        block_solve_reshaped = torch.reshape(Block_Solve,(batch,xt_dim*time))
        self.assertTrue(torch.allclose(torch_inverse_solve,block_solve_reshaped))

    def test_Block_Triangular_Solve_dimensions(self):
        """
        tests the block triangular solve and compares it to fully dense inverse solve implement in torch.
        """
        xt_dim = 3
        time = 4
        batch = 2
        L = torch.rand((time, xt_dim, xt_dim))
        D = torch.exp(L @ torch.transpose(L, dim0=1, dim1=2)) + 5
        B = torch.rand((time - 1, xt_dim, xt_dim))

        # Need to construct cholesky decomposition to get diagonal matrix that is lower-triangular
        D, B = Compute_Block_Cholesky(D, B)
        full_vec = torch.rand((batch, time, xt_dim))
        # Computing block inverse
        Block_Solve = Block_Triangular_Solve(D, B, full_vec)
        self.assertTrue(tuple(Block_Solve.shape) ==(batch,time,xt_dim))