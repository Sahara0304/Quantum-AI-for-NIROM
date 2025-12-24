import numpy as np
import math
import scipy
from scipy.linalg import qr

def centralize(S):
    s_mean=np.average(S,axis=1).reshape(-1,1)
    Sc=S-s_mean
    return Sc, s_mean
def eig(S):
    evals,evecs=np.linalg.eig(S.T@S)
    sorted_indices = np.argsort(evals)
    evals = evals[sorted_indices[::-1]].real
    evecs = evecs[:,sorted_indices[::-1]].real
    return evals, evecs
def proj_op(S,evals,evecs,r):
    Sc,s_mean=centralize(S)
    sorted_indices = np.argsort(evals)
    evals = evals[sorted_indices[::-1]].real
    evecs = evecs[:,sorted_indices[::-1]].real
    Pr=Sc@evecs[:,0:r]@np.diag(1/np.sqrt(evals[0:r]))
    Pr, _ = qr(Pr, mode='economic') 
    evals2=evals
    fid=sum(evals2[0:r])/sum(evals2)
    return Pr,s_mean,fid
def reduce(S,Pr,s_mean):
    Sc=S-s_mean
    u_hat=Pr.T@Sc
    Scr=Pr@u_hat
    Sr=Scr+s_mean
    return Sr, u_hat