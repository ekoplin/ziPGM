#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:46:06 2021

        sample from zero truncated poisson with "mean" param Mu
        
@author: eric
"""
import torch 

def sample_tpoisson(Mu):
    U = torch.rand_like(Mu)# uniform [0,1)
    T =-torch.log(1-U*(1-torch.exp(-Mu)))# time to the first event conditioned to be in (0,Mu)
    T1= Mu-T

    X=torch.ones_like(Mu)
    valid=(T1>0)
    X[valid] += torch.distributions.Poisson(T1[valid]).sample()
    return X
