#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:25:55 2021
        
        Sufficient dimension reduction for zero-inflated graphical models
	Eric Koplin, Liliana Forzani,  Diego Tomassi, Ruth M. Pfeiffer

	Define the kernel regression used to predict a continous outcome based on a reduction
        
@author: eric
"""
import torch

def kernelreg_predict(Rtrain,Ytrain,Rtest):
    median=torch.median(Rtrain.norm(dim=0))

    N=torch.distributions.Normal(0.,median/3)
    pairwise_diffs=Rtest.unsqueeze(2)-Rtrain.unsqueeze(1)#[2, 10, 100]
    pairwise_norm2=pairwise_diffs.norm(dim=0).pow(2)     #[10, 100]
    probs=N.log_prob(pairwise_norm2).exp()
    #yhat=(probs*ytrain[None,:]).sum(1)/probs.sum(1)
    Yhat=Ytrain.mm(probs.t())/probs.sum(1)
    return Yhat
