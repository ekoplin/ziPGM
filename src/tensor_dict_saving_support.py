#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:25:55 2021
        
        Sufficient dimension reduction for zero-inflated graphical models
	Eric Koplin, Liliana Forzani,  Diego Tomassi, Ruth M. Pfeiffer

	save and load parameters in compressed format
        
@author: eric
"""

import pathlib
import numpy as np
import torch

class save(object):
    def __init__(self,path,return_torch_tensor=True):
        if not isinstance(path,pathlib.PurePath):
            path = pathlib.Path(path)
        if not  path.is_dir():
            path.mkdir(parents=True,exist_ok=True)
        self.path          = path
        self.return_torch  = return_torch_tensor
    def save_dict(self,param_dict,name):
        if all([isinstance(v,torch.Tensor) for v in param_dict.values()]):
            param_dict_np = {k:v.cpu().numpy() for k,v in param_dict.items()}
        elif all([isinstance(v,np.ndarray) for v in param_dict.values()]):
            param_dict_np = param_dict
        else:
            raise AssertionError('values known type, all must be or torch.Tensor or np.array')

        # param_cat_np = np.hstack([param_dict_np[k] for k in self.param_names if k in param_dict_np.keys()])
        # np.savetxt(self.path/(name+'.txt'), param_cat_np)
        np.savez_compressed(self.path/(name+'.npz'),**{k:v.astype(np.bool_ if 'vs' in param_dict_np.keys() else np.single) for k,v in param_dict_np.items()})
    def load_dict(self,name):
        # param_cat_np  = np.loadtxt(self.path/(name+'.txt'))
        # p,d           = param_cat_np.shape
        # r             = self.compute_r_from_cat_shape(p,d)
        # param_dict_np = dict(zip(self.param_names,np.hsplit(param_cat_np,np.cumsum(self.param_sizes(p,r)))))
        param_dict_file=np.load(self.path/(name+'.npz'))
        param_dict_np = {k:v.astype(np.bool_ if 'vs' in param_dict_file.files else np.double) for k,v in param_dict_file.items()}
        
        if self.return_torch:
            param_dict = {k:torch.tensor(v) for k,v in param_dict_np.items()}
        else:
            param_dict = param_dict_np
        return param_dict
    def is_file(self,name):
        return (self.path/(name+'.npz')).is_file()
