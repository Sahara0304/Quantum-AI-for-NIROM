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

class QSVD:
    def __init__(self, matrix):
        assert np.allclose(matrix, matrix.T), "Input matrix must be symmetric"
        self.matrix = qnp.array(matrix, requires_grad=False)
        self.n = matrix.shape[0]
        self.num_qubits = int(np.log2(self.n))
        self.params = None
        self.evals = None
        self.evecs = None
        
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        
        @qml.qnode(self.dev, interface="torch")#, diff_method="parameter-shift")
        def circuit(theta, state):
            self.quantum_circuit(theta, state)
            return qml.expval(qml.Hermitian(self.matrix, wires=range(self.num_qubits))), qml.state()
        
        self.circuit = circuit
        
    def quantum_circuit(self, theta, state):
        qml.BasisState(state, wires=range(self.num_qubits))
        theta = theta.reshape(-1, self.num_qubits)
        n_layers = theta.shape[0]
        for layer in range(n_layers):
            for i in range(self.num_qubits):
                qml.RY(theta[layer, i], wires=i)
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i+1)%self.num_qubits])
    
    def initialize_params(self, n_layers=3):
        scale = np.sqrt(2.0 / (self.num_qubits * n_layers))
        self.params = torch.nn.Parameter(
            torch.randn(n_layers, self.num_qubits) * scale,
            requires_grad=True
        )
    
    def loss_function(self):
        M_rec = torch.zeros([self.n,self.n])
        M = torch.tensor(self.matrix, dtype=torch.float32)
        
        for i in range(self.n):
            state = qnp.array([int(b) for b in f"{i:0{self.num_qubits}b}"])
            eval, evec = self.circuit(self.params, state)
            M_rec = M_rec + eval * evec.reshape(-1,1).conj() @ evec.reshape(1,-1)
        loss = torch.norm(M - M_rec, p = 2)
        return loss 

    def optimize(self, iterations=200, lr=0.1):
        optimizer = torch.optim.Adam([self.params], lr=lr)
        loss_history = []
        progress = tqdm(range(iterations))
        for epoch in progress:
            optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            progress.set_description(f"Loss {loss}")
        self.post_process()
        return loss_history
    
    def post_process(self):
        with torch.no_grad():
            evals = qnp.zeros(self.n)
            evecs = qnp.zeros((self.n, self.n))
            
            for i in range(self.n):
                state = qnp.array([int(b) for b in f"{i:0{self.num_qubits}b}"])
                output = self.circuit(self.params, state)
                evals[i] = output[0]
                evecs[:, i] = output[1].real.numpy()

            sorted_indices = np.argsort(evals)[::-1]
            self.evals = evals[sorted_indices]
            self.evecs = evecs[:, sorted_indices]
    
    def get_state_vector(self, state):
        @qml.qnode(self.dev, interface="torch")
        def state_circuit():
            #qml.BasisState(state, wires=range(self.num_qubits))
            self.quantum_circuit(self.params, state)
            return qml.state()
        
        return state_circuit().numpy()