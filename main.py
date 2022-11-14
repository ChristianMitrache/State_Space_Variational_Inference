import Poisson_LDS
import Trainer
from Approximate_Posterior import Time_Series_Approx_Gaussian_First_Param
import torch
import torch.nn as nn
import torch.optim as optim
from Poisson_LDS import Poisson_LDS_Expected_Likelihood
import Mean_Cov_Models
from matplotlib.pyplot import plot
import numpy as np

if __name__ == "__main__":
    # Constructing Random True PLDS:
    xt_dim = 30
    zt_dim = 10
    time = 200
    # instantiating some random true parameters:
    true_A = torch.rand((zt_dim,zt_dim))
    true_C = torch.rand((xt_dim,zt_dim))
    true_Q = torch.rand((zt_dim,zt_dim))
    true_Q = true_Q.T @ true_Q/true_Q.shape[0]
    true_Q = torch.tril(true_Q,diagonal=-1) + \
            torch.diag(torch.exp(torch.diag(true_Q))) + torch.tril(true_Q,diagonal=-1).T
    print(true_Q.shape)
    true_d = torch.rand(xt_dim)
    # sampling from true Poisson LDS model.
    observed_time_series = Poisson_LDS.sample_from_Poisson_LDS(true_Q,true_A,true_C,true_d,time)

    # instantiating random initialization of a poisson LDS model to be trained
    PLDS_model = Poisson_LDS_Expected_Likelihood(xt_dim, zt_dim)

    # Constructing mean-covariance NNs
    lin_layer_dims = (40,50,60)
    non_lin_module = nn.ReLU()
    mean_model = Mean_Cov_Models.Fully_Connected_Mean_Model(xt_dim,zt_dim,lin_layer_dims,non_lin_module)
    inv_cov_model = Mean_Cov_Models.Inverse_Variance_Model(xt_dim,zt_dim,lin_layer_dims,lin_layer_dims,non_lin_module)

    #Iniatilizing approximate posterior loss function:
    Approx_model = Time_Series_Approx_Gaussian_First_Param(mean_model,inv_cov_model)
    KL_Divergence = Trainer.Approximate_Inference_Loss(Approx_model,PLDS_model)

    # hyper-parameters for training:
    batch_size = 100
    num_epochs = 100
    num_samples = 100
    lr = 1e-4

    # Setting Device (GPU/CPU)
    if torch.cuda.is_available():
        device = "cuda"
        print("cuda gpu used")
    else:
        device = "cpu"
        print("cpu is being used")

    # Initialization of data loaders+ optimizers + schedulers for optimization
    time_series_loader = Trainer.Create_Time_Series_Data_Loader(observed_time_series, batch_size)
    KL_Divergence.to(device)
    # optimizer = optim.SGD(model.parameters(),lr = lr_sgd,momentum = 0.85)
    optimizer = optim.Adam(KL_Divergence.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Training Loop:
    training_losses = []
    for i in range(0, num_epochs):
        print("Epoch:")
        print(i)
        print("batch loss at end of epoch:")
        last_batch_loss = Trainer.train_model_epoch(time_series_loader, KL_Divergence, optimizer, scheduler,
                                                    num_samples=num_samples).item()
        print(last_batch_loss)
        training_losses.append(last_batch_loss)

    plot(np.arrange(start =1,stop = num_epochs,step = 1),training_losses)