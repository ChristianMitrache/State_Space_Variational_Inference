import torch
from torch.nn import Module
from Approximate_Posterior import Approximate_Model
from torch.utils.data import Dataset,BatchSampler,SequentialSampler,TensorDataset,DataLoader
import math
import lightning

"""
This file contains classes for loss functions, data set loaders and the training function.
"""

class Approximate_Inference_Loss(Module):

    def __init__(self, Approx_Model: Approximate_Model, User_Model: Module):
        super().__init__()
        self.Approx_Post = Approx_Model
        self.User_model = User_Model

    def forward(self,x,n):
        """
        Makes calls to approximate model class and estimates the KL divergence between the user model and
        and the approximate model.
        :param n: number of samples to take when estimating expectation
        :param x: observations of the time series (a tensor of shape batch_time x x_dim)
        :return: returns an estimate of the ELBO.
        """
        # Note: Sampling needs to be called after forward pass of approximate posterior!
        return -(self.Approx_Post(x) + self.User_model(x,self.Approx_Post.Sample(n, x)))


# Dataloader for getting subsets of original time series
# in time for the original observational time series x (of size time x xt_dim)

# Code below taken from torch documentation:
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset

def Create_Time_Series_Data_Loader(X:torch.Tensor,batch_size:int,num_workers = 0,
                                   persistent_workers = False,pin_memory = True):
    """
    Creates tensor
    :param X: Tensor with all of the data. (In future will change this to class which reads parquet data)
    :param batch_size: batch size.
    :return: returns a PyTorch Dataloader instance
    """
    Tensor_data = TensorDataset(X)
    batched_time_sampler = BatchSampler(SequentialSampler(Tensor_data), batch_size=batch_size, drop_last=False)
    return DataLoader(Tensor_data,batch_sampler=batched_time_sampler,num_workers = num_workers,
                      persistent_workers = persistent_workers, pin_memory = pin_memory)


def train_model_epoch(loader,loss_fn,optimizer,scheduler,
                      num_samples = 100,double = False,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
  """Train model for one epoch and return loss"""

  for batch_index, data in enumerate(loader):
      # Moving Data to GPU (and converting ints to floating points for training)
      if double:
        data = data[0].double().to(device = device)
      else:
          data = data[0].float().to(device=device)
      # Forward pass:
      optimizer.zero_grad()
      loss = loss_fn(data, num_samples)

      # zero out gradients and perform backward pass.
      loss.backward()
      # Updating the weights
      optimizer.step()
      # print("BATCH:")
      # print(batch_index+1)
      print("model_loss for batch")
      print(loss.item())

  scheduler.step() # change learning rate after epoch with scheduler
  return loss