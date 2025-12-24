import numpy as np
import math
import pennylane as qml
import scipy
import meshio
import os
import itertools
import matplotlib.pyplot as plt
from scipy.linalg import qr
from pennylane import numpy as qnp

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

from tqdm import tqdm
class QuantumFeatureExtractor(torch.nn.Module):
    def __init__(self, dim=1, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        self.qlayer = self._build_quantum_layer()
        with torch.no_grad():
            self.qlayer.weights.data.uniform_(-np.pi/8, np.pi/8)  
        self.pre_net = torch.nn.Sequential(
            torch.nn.Linear(dim, n_qubits)
        )

    def _build_quantum_layer(self):
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
                    
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    #qml.Rot(*weights[l, q], wires=q)
                    qml.RY(weights[l, q, 0], wires=q)
                    qml.RZ(weights[l, q, 1], wires=q)   
                for q in range(self.n_qubits):
                    qml.CZ(wires=[q, (q+1)%self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes={"weights": (self.n_layers, self.n_qubits, 2)}
        )
    
    def forward(self, x):
        y = torch.tanh(self.pre_net(x))*torch.pi
        quantum_features = self.qlayer(y)
        return quantum_features#torch.mul(quantum_features,x)

class QuantumDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dim):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = QuantumFeatureExtractor(dim=dim)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=1)#self.feature_extractor.n_qubits)
        )
    
    def forward(self, x):
        #print(x.size[1])
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class qdkl:
    def __init__(self,u_hat):
        self.data=u_hat.T
        self.nt=len(self.data)
        self.x_train=torch.tensor(self.data[0:-1],dtype= torch.float32)
        self.n=self.data.shape[1]
        self.model=[]
        self.likelihood=[]
        self.losses=[]
        self.u_hat_p=None
    def train(self,n_epochs=300):    
        x_train=self.x_train
        data=self.data
        for i in range(self.n):
            y_train=torch.tensor(data[1::,i].reshape(-1,1),dtype=torch.float32).squeeze(1)
            modeli, likelihoodi, lossesi = train_model(x_train,y_train, n_epochs=n_epochs, dim=self.n)
            self.model.append(modeli)
            self.likelihood.append(likelihoodi)
            self.losses.append(lossesi)
    def pred(self,level):
        n=self.n
        u_hat0=self.data[self.nt-1].reshape(1,n)
        u_hat_p=torch.tensor(u_hat0,dtype=torch.float)
        uold=u_hat_p
        likelihood=self.likelihood
        model=self.model
        progress = tqdm(range(level+1-self.nt))
        for epoch in progress:
            yp=torch.zeros(1,n)
            for j in range(n):
                model[j].eval()
                likelihood[j].eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    preds = likelihood[j](model[j](uold))
                    ypj = preds.mean
                yp[0,j]=ypj
            uold=yp
            u_hat_p=torch.cat([u_hat_p[0],uold[0]]).reshape(1,-1)
        self.u_hat_p=np.reshape(u_hat_p,[-1,n]).T
        progress.set_description(f"level {epoch+1}")
    def plot(self):
        plt.figure(figsize=(8, 4))
        for i in range(self.n):
            plt.plot(self.losses[i])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Convergence")
        plt.grid(alpha=0.3)
        plt.show()

def train_model(X_train, y_train, dim, n_epochs=150):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = QuantumDKL(X_train, y_train, likelihood, dim=dim)
    
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 0.05, 'amsgrad': True},
        {'params': model.covar_module.parameters(), 'lr': 0.02},
        {'params': likelihood.parameters(),'lr': 0.02}
    ], weight_decay=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.75,min_lr=0.01)
    
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    progress = tqdm(range(n_epochs))
    for epoch in progress:
    #for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        losses.append(loss.item())
        progress.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
    
    return model, likelihood, losses