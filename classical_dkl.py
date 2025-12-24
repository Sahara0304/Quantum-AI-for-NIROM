import meshio
import os
import itertools
import numpy as np
import math
import scipy
from scipy.linalg import qr
from tqdm import tqdm

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gpytorch
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel
from sklearn.model_selection import train_test_split

class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[32, 64], output_dim=10):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        #layers.append(nn.Tanh())  
        
        self.net = nn.Sequential(*layers)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        return self.net(x)


class ClassicalDKL(ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim=1, 
                 feature_dim=10, hidden_dims=[32, 64]):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = ClassicalFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=feature_dim
        )
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=feature_dim)
        )
    
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_dkl(model, likelihood, train_x, train_y, 
                               epochs=100, lr=0.01, verbose=True):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.covar_module.parameters(), 'lr': lr * 0.5},
        {'params': model.mean_module.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)

    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    
    progress = tqdm(range(epochs))
    for epoch in progress:
        optimizer.zero_grad()
        with gpytorch.settings.cholesky_jitter(1e-4):
            output = model(train_x)
            loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        progress.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
    return model, likelihood, losses


def predict(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
    return mean, lower, upper

class cdkl:
    def __init__(self,u_hat):
        self.data=u_hat.T
        self.n=self.data.shape[1]
        self.model=[]
        self.likelihood=[]
        self.losses=[]
        self.u_hat_p=None
    def train(self,n_epochs=300):    
        data=self.data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        for i in range(self.n):
            y_train=torch.tensor(data[1::,i].reshape(-1,1),dtype=torch.float32).squeeze(1)
            x_train=torch.tensor(data[0:-1,:],dtype= torch.float32)
            modeli = ClassicalDKL(
                train_x=x_train, 
                train_y=y_train,
                likelihood=likelihood,
                input_dim=x_train.size(1),
                feature_dim=4,
                hidden_dims=[4]
            )
            trained_modeli, trained_likelihoodi, lossesi = train_dkl(
                model=modeli,
                likelihood=likelihood,
                train_x=x_train, 
                train_y=y_train,
                epochs=n_epochs,
                lr=0.002
            )
            self.model.append(trained_modeli)
            self.likelihood.append(trained_likelihoodi)
            self.losses.append(lossesi)
    def pred(self,initial_level,level):
        n=self.n
        u_hat0=self.data[initial_level].reshape(1,n)
        u_hat_p=torch.tensor(u_hat0,dtype=torch.float)
        uold=u_hat_p
        likelihood=self.likelihood
        model=self.model
        progress = tqdm(range(level-initial_level))
        for epoch in progress:
            yp=torch.zeros(1,n)
            for j in range(n):
                model[j].eval()
                likelihood[j].eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(5e-4):
                    preds = likelihood[j](model[j](uold))
                    ypj = preds.mean
                yp[0,j]=ypj
            uold=yp
            u_hat_p=torch.cat([u_hat_p[0],uold[0]]).reshape(1,-1)
        self.u_hat_p=np.reshape(u_hat_p,[-1,n]).T
        progress.set_description(f"level {level+epoch+1}")
    def plot(self):
        plt.figure(figsize=(8, 4))
        for i in range(self.n):
            plt.plot(self.losses[i])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Convergence")
        plt.grid(alpha=0.3)
        plt.show()