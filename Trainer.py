import torch
from torch.nn import Module
import Approximate_Posterior
import Poisson_LDS
import User_Model
class Approximate_Inference_Trainer(Module):

    def __init__(self, Approx_Post: Approximate_Posterior, User_Model: Module):
        super().__init__()
        self.Approx_Post = Approx_Post
        self.User_model = User_Model

    def compute_KL_div(self,x):
        self.User_model(x)


