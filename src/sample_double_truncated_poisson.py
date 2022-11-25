#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:46:06 2021

        sample from zero truncated poisson with upper bound Tstar having "mean" param Mu
        
@author: eric
"""
import torch
from poissinv_cpu import poissinv as poisson_inverse_cdf_cpu
try:
    from poissinv_cuda import poissinv as poisson_inverse_cdf_cuda
except:
    extpoisson_inverse_cdf_cuda_cuda=None
def sample_tTpoisson(Mu,Tstar=1e3):# sample double truncated poisson, with lower truncation=1
    # Mu is eta.exp() the mean parameter
    # U in [0,1], 
    # if parallel_var, paralelize across p dimension, otherwise paralelize across samples
    b=torch.igammac((Tstar if isinstance(Tstar,torch.Tensor) else torch.tensor(Tstar))+1,Mu)
    a=torch.igammac(torch.tensor(0)+1,Mu)
    # must samlpe a unifiform [a,b]
    U=torch.rand_like(Mu)*(b-a)+a
    
    shape=Mu.size()
    if Mu.is_cuda:
        if Mu.dim()==2:
            X=poisson_inverse_cdf_cuda(Mu,U)
        elif Mu.dim()==1:
            X=poisson_inverse_cdf_cuda(Mu.view(-1,1),U.view(-1,1))
        else:
            raise AssertionError('Mu must be a unidimensional if bidimensional tensor')
    else: 
        if Mu.dim()==2:
            X=poisson_inverse_cdf_cpu(Mu.view(-1),U.view(-1))
        elif Mu.dim()==1:
            X=poisson_inverse_cdf_cpu(Mu,U)
        else:
            raise AssertionError('Mu must be a unidimensional if bidimensional tensor')
    return X.view(shape)

def sample_Tpoisson(Mu,Tstar=1e3):# sample upper truncated poisson
    # Mu is eta.exp() the mean parameter
    # U in [0,1], 
    # if parallel_var, paralelize across p dimension, otherwise paralelize across samples
    b=torch.igammac((Tstar if isinstance(Tstar,torch.Tensor) else torch.tensor(Tstar))+1,Mu)
    # must samlpe a unifiform [0,b]
    U=torch.rand_like(Mu)*b
    
    shape=Mu.size()
    if Mu.is_cuda:
        if Mu.dim()==2:
            X=poisson_inverse_cdf_cuda(Mu,U)
        elif Mu.dim()==1:
            X=poisson_inverse_cdf_cuda(Mu.view(-1,1),U.view(-1,1))
        else:
            raise AssertionError('Mu must be a unidimensional if bidimensional tensor')
    else: 
        if Mu.dim()==2:
            X=poisson_inverse_cdf_cpu(Mu.view(-1),U.view(-1))
        elif Mu.dim()==1:
            X=poisson_inverse_cdf_cpu(Mu,U)
        else:
            raise AssertionError('Mu must be a unidimensional if bidimensional tensor')
    return X.view(shape)
