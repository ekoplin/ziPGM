#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:20:07 2021
        
        Sufficient dimension reduction for zero-inflated graphical models
	Eric Koplin, Liliana Forzani,  Diego Tomassi, Ruth M. Pfeiffer
	
	Optimization methods for penalized and reffiting problems for PGM family
	
@author: eric
"""
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

import time
import torch
from copy import deepcopy
import pathlib
import json
from torch.utils.tensorboard import SummaryWriter

### weight estimation ###

class PGM_hessian_fisher(object):
    '''
    when sample=True and the model is independet, compute the Fisher
    when sample=False, compute the Hessian 
    '''
    @property
    def eigen_max(self):
        inter_aux=self.eigenvalues['inter'].clone()
        p=inter_aux.size(0)
        inter_aux[range(p),range(p),:]=0
        inter_max=inter_aux.max()
        return max([self.eigenvalues[k].max() if k!='inter' else inter_max for k in self.eigenvalues.keys()])
    @property
    def eigen_min(self):
        inter_aux=self.eigenvalues['inter'].clone()
        p=inter_aux.size(0)
        inter_aux[range(p),range(p),:]=torch.finfo(torch.get_default_dtype()).max
        inter_min=inter_aux.min()
        return min([self.eigenvalues[k].min() if k!='inter' else inter_min for k in self.eigenvalues.keys()])
    @property
    def condition_number(self):
        return self.eigen_max/self.eigen_min
    def __init__(self,model,param_dict,X,Y,damping=1e-6,mineig=1e-30,maxeig=1e30,fisher=False,**kwargs):
        assert X.size(1)==Y.size(1),'n'
        assert mineig>0,'positive constant'
        
        self.model      = model
        self.fisher     = fisher
        self.kwargs     = kwargs

        self.damping    = damping
        self.mineig     = mineig
        self.maxeig     = maxeig

        self.Y = Y 
        self.X = X
                         
        self.eigenvalues,self.eigenvectors = self.process_hessian(param_dict)
    def update(self,param_dict,eps=0):
        # eps constrols the memory, if it is 0, forget about the past, if it is 1, do not update
        if eps<1:
            # new = self.__class__(model,param_dict,X,H,Y,damping,mineig,maxeig,ell_normalized_by_p)
            eigenvalues,eigenvectors=self.process_hessian(param_dict)
            if eps==0:
                self.eigenvalues,self.eigenvectors=eigenvalues,eigenvectors
            else:
                # combine
                Hold      = self.apply(lambda x:x)
                Hnew      = {k:eigenvectors[k] @ torch.diag_embed(eigenvalues[k]) @ eigenvectors[k].transpose(-2, -1) for k in ['inter','reg','lin']}
                Hcomb     = {k:eps*Hold[k]+(1-eps)*Hnew[k] for k in Hold.keys()}
                Lint,Qint = torch.linalg.eigh(Hcomb['inter'])
                Lreg,Qreg = torch.linalg.eigh(Hcomb['reg'])
                Llin,Qlin = torch.linalg.eigh(Hcomb['lin'])
                self.eigenvalues  = dict(
                                        inter = Lint,
                                        reg   = Lreg,
                                        lin   = Llin
                                        )
                self.eigenvectors = dict(
                                        inter = Qint,
                                        reg   = Qreg,
                                        lin   = Qlin
                                        )
    def process_hessian(self,param_dict):
        '''   
        regularization can be chosen to enforce definite positiveness by https://math.stackexchange.com/questions/2587314/eigenvalues-of-the-addition-of-a-symmetric-matrix-and-the-identity-matrix
        https://hal.inria.fr/hal-01552340v1/document
        '''
        if self.fisher:# check indep
            # assert all([torch.allclose(v,torch.zeros(1),atol=1e-99) for k,v in param_dict.items() if k not in ('eta','xi','ThetaDiag')]),'fisher can only be computed at independence'
            X = self.model.sample(self.Y,param_dict,X0=self.X,**self.kwargs)
        else:
            X = self.X
        HH = self.model.cat_hessian(X, self.Y, param_dict,symmetric=True)

        # add damping
        for v in HH.values():
            v_diag=torch.diagonal(v,dim1=-1,dim2=-2)
            v_diag+=self.damping

        # eigen
        Lint,Qint = torch.linalg.eigh(HH['inter'])
        Lreg,Qreg = torch.linalg.eigh(HH['reg'])
        Llin,Qlin = torch.linalg.eigh(HH['lin'])
        
        for L in [Lint,Lreg,Llin]:
            # L+=self.damping
            L_min=L.min()
            if L_min<self.mineig:
                logger.warning(f'eigen shrunken because L_min={L_min}<{self.mineig}')
                L+=self.mineig-L_min
            L_max=L.max()
            if L_max>self.maxeig:
                logger.warning(f'eigen shrunken because L_max={L_max}>{self.maxeig}')
                L.clamp_max(self.maxeig)

        return  dict(inter=Lint,reg=Lreg,lin=Llin),dict(inter=Qint,reg=Qreg,lin=Qlin)
    
    def apply(self,f):
        # apply a function to the matrix (by transforming the eigenvalues)
        return {k:self.eigenvectors[k] @ torch.diag_embed(f(self.eigenvalues[k])) @ self.eigenvectors[k].transpose(-2, -1) for k in ['inter','reg','lin']}
    def apply_named(self,**f):
        # apply a function to the eigenvalues depending on group
        return {k:self.eigenvectors[k] @ torch.diag_embed(f[k](self.eigenvalues[k])) @ self.eigenvectors[k].transpose(-2, -1) for k in ['inter','reg','lin']}
    @classmethod
    def product(cls,pF,x):
        # returns HF @ x
        return {k:torch.matmul(pF[k],x[k]) for k in x.keys()}
    @classmethod
    def product_of_differences(cls,pF,x,z):
        # returns HF@(x-z)
        return {k:torch.matmul(pF[k],x[k]-z[k]) for k in x.keys()}
    @classmethod 
    def sum(cls,*dicts):
        # sum of dictionaries
        result={}
        for d in dicts:
            for k,v in d.items():
                if k in result.keys():
                    result[k]+=v
                else:
                    result[k]=v.clone()
        return result    

### pnopt aux ###

def cat_params(param_dict):
    # group according block structure, used to multiply with compute_preconditioning_matrices
    cat={}
    if 'Theta' in param_dict.keys():
        cat['inter']=param_dict['Theta'].unsqueeze(-1).unsqueeze(-1)#pxpx4x1
    if 'Gamma' in param_dict.keys():
        cat['reg']=param_dict['Gamma'].unsqueeze(-1)#px2rx1
    if 'eta' in param_dict.keys():
       cat['lin']=torch.cat([param_dict[k] for k in (('eta','ThetaDiag') if 'ThetaDiag' in param_dict.keys() else ('eta',))],1).unsqueeze(-1)#px2x1
    return cat
def uncat_params(cat_param_dict):
    # group according block structure, used to multiply with compute_preconditioning_matrices
    param={}
    if 'inter' in cat_param_dict.keys():
        param['Theta']=cat_param_dict['inter'].squeeze(-1).squeeze(-1)#pxp,pxp,pxp,pxp
    if 'reg' in cat_param_dict.keys():
        param['Gamma']=cat_param_dict['reg'].squeeze(-1)#pxr,pxr
    if 'lin' in cat_param_dict.keys():
        if cat_param_dict['lin'].size(1)==1:
            param['eta']   =cat_param_dict['lin'].squeeze(-1)#px1,px1
        elif cat_param_dict['lin'].size(1)==2:
            param['eta'],param['ThetaDiag']=torch.split(cat_param_dict['lin'].squeeze(-1),1,1)#px1,px1
        else:
            raise AssertionError('size missmatch')
    return param

def prox_inter(y_inter,threshold=0):
    # threshold: regularization parameter
    # y_inter ~ pxpx4x1
    assert y_inter.size(0)==y_inter.size(1),'p'
    group_soft = 1-threshold/y_inter.norm(dim=2,keepdim=True)
    group_soft[group_soft!=group_soft]=0 # avoid nans
    return torch.clamp(group_soft,min=0)*y_inter#=torch.threshold(group_soft,0,0)*A#=torch.max(group_soft,torch.zeros(A.size()))*A

def prox_reginter(y_reg,y_inter,threshold=0):
    # y_reg   ~ pxrx1
    # y_inter ~ pxpx4x1
    # consider the groups associated to (Gamma_j,(Theta_{jl})_{l=1,...,k})
    group_norm = (y_inter.norm(dim=(1,2),keepdim=True).pow(2)+y_reg.norm(dim=(1),keepdim=True).pow(2).unsqueeze(-1)).sqrt()
    group_soft = 1-threshold/group_norm
    group_soft[group_soft!=group_soft]=0 # avoid nans
    return torch.clamp(group_soft,min=0).squeeze(-1)*y_reg,torch.clamp(group_soft,min=0)*y_inter

### pnopt with anysometric hierarchical penalization for HMs ###

def subproblemSDMM( model,x,dx,FisherObj,FisherPF,t,
                    lambda_reginter=0,lambda_inter=0,
                    step=1e-5,# gamma
                    max_iter=100,eps_conv=1e-9,eps_feas=1e-4,
                    summary_writer=None
                    ):   
    '''
    SDMM algorihtm from [Combettes 2011].
    '''
    #x0=x
    smax  = torch.cat([v.reshape(-1) for v in FisherObj.eigenvalues.values()]).median().sqrt() #torch.ones(1)#max([v.sqrt().max() for v in pseudoFisherObj.eigenvalues.values()])
    # inter: loss(Fsqx), ic(smaxIx), reg1(PFsqx), reg2(PFsqx) 
    # reg  : loss(Fsqx), ic(smaxIx), reg1(PFsqx)
    # lin  : loss(Fsqx), ic(smaxIx)  
    PFPF=FisherPF.apply_named(**{'inter':lambda x: 2*x,   
                                  'reg' :lambda x: 1*x,   
                                  'lin' :lambda x: 0*x})  
    FFpII=FisherObj.apply_named(**{'inter':lambda x: x+smax.pow(2),   
                                    'reg' :lambda x: x+smax.pow(2),   
                                    'lin' :lambda x: x+smax.pow(2)})  
    Qinv = {k:torch.linalg.inv(PFPF[k]+FFpII[k]) for k in x.keys()}
    
    # transformations
    PFsq   = FisherPF.apply(lambda x: torch.sqrt(x))
    Hsq    = FisherObj.apply(lambda x: torch.sqrt(x))
    
    # part of the closed form proxiaml loss
    Hsqinv= FisherObj.apply(lambda x: 1/torch.sqrt(x))
    HsqX  = FisherObj.product(Hsq,x)
    HisqdX= FisherObj.product(Hsqinv,dx)
    HxHdx = {k:HsqX[k]/t-HisqdX[k] for k in x.keys()}
    
    # init
    y_loss    =FisherObj.product(Hsq,x)
    y_const   ={k:smax*x[k] for k in x.keys()}
    y_reginter=FisherPF.product(PFsq,{k:v for k,v in x.items() if k in ['inter','reg']})
    y_inter   ={k:y_reginter[k].clone() for k in ['inter']}
    
    z_loss    = {k:torch.zeros_like(v) for k,v in y_loss.items()}
    z_const   = {k:torch.zeros_like(v) for k,v in y_const.items()}
    z_reginter= {k:torch.zeros_like(v) for k,v in y_reginter.items()}
    z_inter   = {k:torch.zeros_like(v) for k,v in y_inter.items()}
    
    
    for it in range(max_iter):
        x_sum = FisherObj.sum(
                    FisherObj.product_of_differences(Hsq,y_loss,z_loss),
                    {k:smax*(y_const[k]-z_const[k]) for k in y_const.keys()},
                    FisherPF.product_of_differences(PFsq,y_reginter,z_reginter),
                    FisherPF.product_of_differences(PFsq,y_inter,z_inter)
                    )
        
        x = FisherObj.product(Qinv,x_sum)
        
        s_loss    =FisherObj.product(Hsq,x)
        s_const   ={k:smax*x[k] for k in x.keys()}
        s_reginter=FisherPF.product(PFsq,{k:v for k,v in x.items() if k in y_reginter.keys()})
        s_inter   ={k:y_reginter[k].clone() for k in ['inter']}
          
        y_loss    ={k:(HxHdx[k]+(s_loss[k]+z_loss[k])/step)*1/(1/t+1/step) for k in y_loss.keys()}
        
        y_const_uncat = uncat_params({k:s_const[k]+z_const[k] for k in y_const.keys()})
        y_const   = cat_params({**model.project_interacts(**{k:y_const_uncat[k] for k in ['Theta','ThetaDiag']  if k in y_const_uncat.keys()}),
                            **model.project_regression(**{k:y_const_uncat[k] for k in ['Gamma']}),
                            **{k:y_const_uncat[k] for k in ['eta']}
                            })
        y_reginter= {**dict(zip(('reg','inter'),(prox_reginter(s_reginter['reg']+z_reginter['reg'],s_reginter['inter']+z_reginter['inter'],step*lambda_reginter))))}
        y_inter   = {'inter':prox_inter(s_inter['inter']+z_inter['inter'],step*lambda_inter)}
        
        z_loss = {k:z_loss[k]+s_loss[k]-y_loss[k] for k in y_loss.keys()}
        z_const = {k:z_const[k]+s_const[k]-y_const[k] for k in y_const.keys()}
        z_reginter = {k:z_reginter[k]+s_reginter[k]-y_reginter[k] for k in y_reginter.keys()}
        z_inter = {k:z_inter[k]+s_inter[k]-y_inter[k] for k in y_inter.keys()}
        
        if summary_writer is not None and not it%(max_iter//10):
            # summary_writer.add_scalar('normZ',sum([zi[k].pow(2).sum() for zi in (z_loss,z_const,z_reginter,z_inter) for k in zi.keys()]).sqrt(),it)
            aux={k:s_loss[k]/t-HxHdx[k] for k in s_loss.keys()}
            q  =sum([v.pow(2).sum()*t/2 for v in aux.values()])
            
            reginter_norm=(s_reginter['inter'].norm(dim=(1,2),keepdim=True).pow(2)+s_reginter['reg'].norm(dim=(1),keepdim=True).pow(2).unsqueeze(-1)).sqrt().sum()
            inter_norm=s_reginter['inter'].norm(dim=2).sum()
            g= lambda_reginter*reginter_norm+lambda_inter*inter_norm
            summary_writer.add_scalar('pnopt_subproblem/loss',q+g,it)

        pattern={} 
        pattern['vs']=y_reginter['reg'].bool().any(1).any(1)
        # simetrize by and rule cases of bad convergence
        pattern['ci']=torch.bitwise_and(torch.outer(pattern['vs'].squeeze(),pattern['vs'].squeeze()),
                                    y_inter['inter'].bool().any(2).squeeze(2).bitwise_and(y_inter['inter'].bool().any(2).squeeze(2).t())) 
    return x,pattern
def subproblemSDMM_inter( model,x,dx,FisherObj,FisherPF,t,
                    lambda_inter=0,
                    step=1e-5,# gamma
                    max_iter=100,eps_conv=1e-9,eps_feas=1e-4,
                    summary_writer=None
                    ):   
    '''
    SDMM algorihtm from [Combettes 2011].
    '''
    #x0=x    
    smax  = torch.cat([v.reshape(-1) for v in FisherObj.eigenvalues.values()]).median().sqrt() #torch.ones(1)#max([v.sqrt().max() for v in pseudoFisherObj.eigenvalues.values()])
    # inter: loss(Fsqx), ic(smaxIx), reg1(PFsqx), reg2(PFsqx) 
    # reg  : loss(Fsqx), ic(smaxIx), reg1(PFsqx)
    # lin  : loss(Fsqx), ic(smaxIx)  
    PFPF=FisherPF.apply_named(**{'inter':lambda x: 1*x,   
                                  'reg' :lambda x: 0*x,   
                                  'lin' :lambda x: 0*x})  
    FFpII=FisherObj.apply_named(**{'inter':lambda x: x+smax.pow(2),   
                                    'reg' :lambda x: x+smax.pow(2),   
                                    'lin' :lambda x: x+smax.pow(2)})  
    Qinv = {k:torch.linalg.inv(PFPF[k]+FFpII[k]) for k in x.keys()}
    
    # transformations
    PFsq   = FisherPF.apply(lambda x: torch.sqrt(x))
    Hsq    = FisherObj.apply(lambda x: torch.sqrt(x))
    
    # part of the closed form proxiaml loss
    Hsqinv= FisherObj.apply(lambda x: 1/torch.sqrt(x))
    HsqX  = FisherObj.product(Hsq,x)
    HisqdX= FisherObj.product(Hsqinv,dx)
    HxHdx = {k:HsqX[k]/t-HisqdX[k] for k in x.keys()}
    
    # init
    y_loss    =FisherObj.product(Hsq,x)
    y_const   ={k:smax*x[k] for k in x.keys()}
    y_inter=FisherPF.product(PFsq,{k:v for k,v in x.items() if k in ['inter']})
    
    z_loss    = {k:torch.zeros_like(v) for k,v in y_loss.items()}
    z_const   = {k:torch.zeros_like(v) for k,v in y_const.items()}
    z_inter   = {k:torch.zeros_like(v) for k,v in y_inter.items()}
    
    
    for it in range(max_iter):
        x_sum = FisherObj.sum(
                    FisherObj.product_of_differences(Hsq,y_loss,z_loss),
                    {k:smax*(y_const[k]-z_const[k]) for k in y_const.keys()},
                    FisherPF.product_of_differences(PFsq,y_inter,z_inter)
                    )
        
        x = FisherObj.product(Qinv,x_sum)
        
        s_loss    =FisherObj.product(Hsq,x)
        s_const   ={k:smax*x[k] for k in x.keys()}
        s_inter=FisherPF.product(PFsq,{k:v for k,v in x.items() if k in y_inter.keys()})
          
        y_loss    ={k:(HxHdx[k]+(s_loss[k]+z_loss[k])/step)*1/(1/t+1/step) for k in y_loss.keys()}
        
        y_const_uncat = uncat_params({k:s_const[k]+z_const[k] for k in y_const.keys()})
        y_const   = cat_params({**model.project_interacts(**{k:y_const_uncat[k] for k in ['Theta','ThetaDiag']  if k in y_const_uncat.keys()}),
                            **model.project_regression(**{k:y_const_uncat[k] for k in ['Gamma']}),
                            **{k:y_const_uncat[k] for k in ['eta']}
                            })
        y_inter   = {'inter':prox_inter(s_inter['inter']+z_inter['inter'],step*lambda_inter)}
        
        z_loss = {k:z_loss[k]+s_loss[k]-y_loss[k] for k in y_loss.keys()}
        z_const = {k:z_const[k]+s_const[k]-y_const[k] for k in y_const.keys()}
        z_inter = {k:z_inter[k]+s_inter[k]-y_inter[k] for k in y_inter.keys()}
        
        if summary_writer is not None and not it%(max_iter//10):
            # summary_writer.add_scalar('normZ',sum([zi[k].pow(2).sum() for zi in (z_loss,z_const,z_reginter,z_inter) for k in zi.keys()]).sqrt(),it)
            aux={k:s_loss[k]/t-HxHdx[k] for k in s_loss.keys()}
            q  =sum([v.pow(2).sum()*t/2 for v in aux.values()])
            
            inter_norm=s_inter['inter'].norm(dim=2).sum()
            g= lambda_inter*inter_norm
            summary_writer.add_scalar('pnopt_subproblem/loss',q+g,it)

        pattern={} 
        pattern['vs']=torch.ones(y_inter['inter'].size(0),1,dtype=torch.bool)
        # simetrize by and rule cases of bad convergence
        pattern['ci']=torch.bitwise_and(torch.outer(pattern['vs'].squeeze(),pattern['vs'].squeeze()),
                                    y_inter['inter'].bool().any(2).squeeze(2).bitwise_and(y_inter['inter'].bool().any(2).squeeze(2).t())) 
    return x,pattern
def PGM_pnopt(*,model,# depends on the model, are methods declared in HM.py
         X,Y,#data
         init_dict,#initial value
         FisherPF,# pseudoFisher object, used in weighted norms
         FisherObj=None,# used as hessian, if none copy FisherPF
         lambda_reginter=0.,lambda_inter=0.,# regularization
         # step=1e-1,# max step size
         err_relP=1e-6,#err_evaluate_each=100,# convergence
         err_relL=1e-6,#err_evaluate_each=100,# convergence
         tol_change_vs=0,# https://nowak.ece.wisc.edu/Wright_Nowak_Figueiredo_2008.pdf
         tol_change_ci=0,
         max_iter=100,# time limit
         precompute_step=False,# if True, ignores step and compute the step by Lee's condition
         alpha=.1,beta=.5,max_rejects=10,# 0<alpha<.5, 0<beta<1, alpha=armijo forcing beta=decreasing step by
         update_hessian_each=10,update_hessian_eps =0,# 0==newton 1==fixed at FisherObj
         summary_writer=None,#evaluate_hat_dict_for_summary_writer=None,# tracing
         subproblemSDMM_kwars={}
         ):
    '''
    pnopt algorithm from [Lee 2012].
    '''
    time_init=time.process_time()
    def h(x,pseudoFisherPF):
        xbar = pseudoFisherPF.product(pseudoFisherPF.apply(lambda x:torch.sqrt(x)),x)
        reginter_norm=(xbar['inter'].norm(dim=(1,2),keepdim=True).pow(2)+xbar['reg'].norm(dim=(1),keepdim=True).pow(2).unsqueeze(-1)).sqrt().sum()
        inter_norm=xbar['inter'].norm(dim=2).sum()
        return lambda_reginter*reginter_norm+lambda_inter*inter_norm
    def ell(param_dict):
        return model.ell(X,Y,param_dict)
   
    if FisherObj is None:
        FisherObj=deepcopy(FisherPF)
    x=cat_params(init_dict)
    x_={k:torch.finfo(torch.get_default_dtype()).max*torch.ones_like(v) for k,v in x.items()}

    pattern_=dict(vs=init_dict['Gamma'].bool().any(1),
                  ci=init_dict['Theta'].bool().bool())
    status = 'max_iter reached'
    for it in range(max_iter):
        param_dict=uncat_params(x)

        if it>0 and not it%update_hessian_each:
            FisherObj.update(param_dict,eps=update_hessian_eps)
            
        dx = model.cat_jacobian(X, Y, param_dict)#cat_params(d_ell(model.ell, X, H, Y, param_dict))
        
        if lambda_reginter==0.:
            z,pattern=subproblemSDMM_inter(model,
                                    x,dx,FisherObj,FisherPF,1,
                                    lambda_inter=lambda_inter,#/FisherPF.eigenvalues['inter'].max(),
                                    **{**dict(max_iter=100,eps_conv=1e-6,eps_feas=1e-6),**subproblemSDMM_kwars},
                                    summary_writer=summary_writer
                                    )    
        else:
            z,pattern=subproblemSDMM(model,
                                    x,dx,FisherObj,FisherPF,1,
                                    lambda_reginter=lambda_reginter,#/FisherPF.eigenvalues['inter'].max(),
                                    lambda_inter=lambda_inter,#/FisherPF.eigenvalues['inter'].max(),
                                    **{**dict(max_iter=100,eps_conv=1e-6,eps_feas=1e-6),**subproblemSDMM_kwars},
                                    summary_writer=summary_writer
                                    )    
        v = {k:z[k]-x[k] for k in x.keys()}
        
        # subproblem_step=subproblemSDMM_kwars.pop('step',1)
        if precompute_step:# LEE2012 satisfies descent condition
            t=min(1,(1-alpha)*2/FisherObj.condition_number)# lee
        else:
            t=1
        
        dxv=sum([(dx[k]*v[k]).sum() for k in dx.keys()])
        regets=0
        gx=ell(param_dict)
        hx=h(x,FisherPF)
        fx=gx+hx
        hz=h(z,FisherPF)
        for regets in range(max_rejects):
            xtv={k:x[k]+t*v[k] for k in x.keys()}
            xtv_param=uncat_params(xtv)
            gxtv=ell(xtv_param)
            hxtv=h(xtv,FisherPF)
            fxtv=gxtv+hxtv
            delta=dxv+hz-hx
            if fxtv<=fx+alpha*t*delta:
                break
            else:
                t*=beta
        
        # print(f'it={it}, update x with t={t}')
        x_=x
        x=xtv
        rel_err =sum([(x[k]-x_[k]).pow(2).sum() for k in x.keys()]).sqrt()/sum([x[k].pow(2).sum() for k in x.keys()]).sqrt()
        rel_loss=(fx-fxtv).abs()/fx.abs()
        if summary_writer is not None:
            # summary_writer.add_scalars('GD_evaluate_hat_dict',evaluate_hat_dict_for_summary_writer({**hat_dict,**nontraindict}),it)
            summary_writer.add_scalar('pnopt/loss',fxtv,it)
            summary_writer.add_scalar('pnopt/relloss',rel_loss,it)
            summary_writer.add_scalar('pnopt/relerr',rel_err,it)
            
            summary_writer.add_scalar('pnopt/pattern/vs',pattern['vs'].sum(),it)
            summary_writer.add_scalar('pnopt/pattern/ci',pattern['ci'].triu(1).sum(),it)

        if rel_err<err_relP or rel_loss<err_relL:
            status = 'converged: err_relP or err_relL reached'
            break
        
        if not (it+1)%update_hessian_each:# in the next step the hessian will be updated
            # compare patterns
            if float(torch.bitwise_xor(pattern['vs'],pattern_['vs']).sum()/max(pattern['vs'].sum(),1))<tol_change_vs and float(torch.bitwise_xor(pattern['ci'],pattern_['ci']).sum()/max(pattern['ci'].sum(),1))<tol_change_ci:
                status = 'converged: tol_change_vs and tol_range_ci reached'
                break
            else:
                pattern_=pattern
        if any([v.isnan().any() for v in x.values()]):
            status = 'nan encountered in x'
            logger.warning(f"pnopt {status} after {it} iterations")
            break
    time_elapsed = time.process_time() - time_init
    if summary_writer is not None:
        summary_writer.add_text('convergence', status, global_step=it)
        summary_writer.add_scalar('time', time_elapsed,global_step=it)
    logger.info(f"pnopt converged after {it} iterations with status {status}")
    return uncat_params(x),pattern

### pnopt with fixed pattern used to refit ###   

def subproblemSDMM_pattern( model,x,dx,pseudoFisherObj,t,
                    pattern_dict,
                    step=.1,# gamma
                    max_iter=100,eps_conv=1e-9,eps_feas=1e-4,
                    summary_writer=None
                    ):
    '''
    SDMM algorihtm from [Combettes 2011].
    '''
    #
    Hsq   = pseudoFisherObj.apply(lambda x: torch.sqrt(x))
    Hsqinv= pseudoFisherObj.apply(lambda x: 1/torch.sqrt(x))
    # smax  = torch.ones(1)#max([v.sqrt().max() for v in pseudoFisherObj.eigenvalues.values()])
    smax  = torch.cat([v.reshape(-1) for v in pseudoFisherObj.eigenvalues.values()]).median().sqrt() #torch.ones(1)#max([v.sqrt().max() for v in pseudoFisherObj.eigenvalues.values()])
    Qinv  = pseudoFisherObj.apply_named(**{'inter':lambda x: 1/(x+2*smax.pow(2)),   # loss(Hsqx), ic(Ix), pattern(Ix)
                                             'reg':lambda x: 1/(x+2*smax.pow(2)),   # loss(Hsqx), ic(Ix), pattern(Ix)
                                             'lin':lambda x: 1/(x+1*smax.pow(2))})  # loss(Hsqx), ic(Ix)
    
    HsqX  = pseudoFisherObj.product(Hsq,x)
    HisqdX= pseudoFisherObj.product(Hsqinv,dx)
    HxHdx = {k:HsqX[k]/t-HisqdX[k] for k in x.keys()}
       
    # init
    y_loss    =pseudoFisherObj.product(Hsq,x)
    y_const   ={k:smax*x[k] for k in x.keys()}
    y_pattern ={k:y_loss[k] for k in ['inter','reg']}
    
    z_loss    = {k:torch.zeros_like(v) for k,v in y_loss.items()}
    z_const   = {k:torch.zeros_like(v) for k,v in y_const.items()}
    z_pattern= {k:torch.zeros_like(v) for k,v in y_pattern.items()}
    
    
    for it in range(max_iter):
        x_sum = pseudoFisherObj.sum(
                    pseudoFisherObj.product_of_differences(Hsq,y_loss,z_loss),
                    {k:smax*(y_const[k]-z_const[k]) for k in y_const.keys()},
                    {k:smax*(y_pattern[k]-z_pattern[k]) for k in y_pattern.keys()},
                    )
        
        x = pseudoFisherObj.product(Qinv,x_sum)
        
        s_loss    =pseudoFisherObj.product(Hsq,x)
        s_const   ={k:smax*x[k] for k in y_const.keys()}
        s_pattern ={k:smax*x[k] for k in y_pattern.keys()}
          
        y_loss    ={k:(HxHdx[k]+(s_loss[k]+z_loss[k])/step)*1/(1/t+1/step) for k in y_loss.keys()}
        y_const_uncat = uncat_params({k:s_const[k]+z_const[k] for k in y_const.keys()})
        y_const   = cat_params({**model.project_interacts(**{k:y_const_uncat[k] for k in ['Theta','ThetaDiag']  if k in y_const_uncat.keys()}),
                            **model.project_regression(**{k:y_const_uncat[k] for k in ['Gamma']}),
                            **{k:y_const_uncat[k] for k in ['eta']}
                            })
        y_pattern = {   'reg'  :(s_pattern['reg']+z_pattern['reg'])*pattern_dict['vs'].squeeze().unsqueeze(-1).unsqueeze(-1),########### cambiar shapeeee
                        'inter':(s_pattern['inter']+z_pattern['inter'])*pattern_dict['ci'].unsqueeze(-1).unsqueeze(-1)}
        
        z_loss    = {k:z_loss[k]+s_loss[k]-y_loss[k] for k in y_loss.keys()}
        z_const   = {k:z_const[k]+s_const[k]-y_const[k] for k in y_const.keys()}
        z_pattern = {k:z_pattern[k]+s_pattern[k]-y_pattern[k] for k in y_pattern.keys()}

        
        if summary_writer is not None and not it%(max_iter//10):
            # summary_writer.add_scalar('normZ',sum([zi[k].pow(2).sum() for zi in (z_loss,z_const,z_reginter,z_inter) for k in zi.keys()]).sqrt(),it)
            aux={k:s_loss[k]/t-HxHdx[k] for k in s_loss.keys()}
            q  =sum([v.pow(2).sum()*t/2 for v in aux.values()])
            summary_writer.add_scalar('pnopt_pattern_subproblem/loss',q,it)
    return x

def PGM_pnopt_pattern(*,model,# depends on the model, are methods declared in HM.py
         X,Y,#data
         init_dict,#initial value
         FisherObj,# used as hessian, if none copy FisherPF
         pattern_dict,
         err_relP=1e-6,err_relL=1e-6,# convergence
         max_iter=100,# time limit
         precompute_step=False,# if True, precompute the step by Lee's condition and enter into the armijo condition
         alpha=.5,beta=.5,max_rejects=10,# 0<alpha<.5, 0<beta<1, alpha:armijo forcing beta:decreasing step by
         update_hessian_each=10,update_hessian_eps =0,# 0==newton 1==fixed at FisherObj
         summary_writer=None,#evaluate_hat_dict_for_summary_writer=None,# tracing
         subproblemSDMM_kwars={}
         ):
    '''
    pnopt algorithm from [Lee 2012].
    '''
    time_init=time.process_time()
    def ell(param_dict):
        return model.ell(X,Y,param_dict)

    x=cat_params(init_dict)
    x_={k:torch.finfo(torch.get_default_dtype()).max*torch.ones_like(v) for k,v in x.items()}
    
    status = 'max_iter reached'
    for it in range(max_iter):
        param_dict=uncat_params(x)

        if it>0 and not it%update_hessian_each:
            FisherObj.update(param_dict,eps=update_hessian_eps)
            
        dx = model.cat_jacobian(X, Y, param_dict)#cat_params(d_ell(model.ell, X, H, Y, param_dict))
        
        z=subproblemSDMM_pattern(model,
                                x,dx,FisherObj,1,
                                pattern_dict,
                                **{**dict(max_iter=100,eps_conv=1e-6,eps_feas=1e-6),**subproblemSDMM_kwars},
                                summary_writer=summary_writer
                                )    
        v = {k:z[k]-x[k] for k in x.keys()}
        
        # subproblem_step=subproblemSDMM_kwars.pop('step',1)
        if precompute_step:# LEE2012 satisfies descent condition
            t=min(1,(1-alpha)*2/FisherObj.condition_number)# lee
        else:
            t=1
        
        dxv=sum([(dx[k]*v[k]).sum() for k in dx.keys()])
        regets=0
        gx=ell(param_dict)
        fx=gx
        for regets in range(max_rejects):
            xtv={k:x[k]+t*v[k] for k in x.keys()}
            xtv_param=uncat_params(xtv)
            gxtv=ell(xtv_param)
            fxtv=gxtv
            delta=dxv
            if fxtv<=fx+alpha*t*delta:
                break
            else:
                t*=beta
        
        # print(f'it={it}, update x with t={t}')
        x_=x
        x=xtv
        rel_err =sum([(x[k]-x_[k]).pow(2).sum() for k in x.keys()]).sqrt()/sum([x[k].pow(2).sum() for k in x.keys()]).sqrt()
        rel_loss=(fx-fxtv).abs()/fx.abs()
        if summary_writer is not None:
            # summary_writer.add_scalars('GD_evaluate_hat_dict',evaluate_hat_dict_for_summary_writer({**hat_dict,**nontraindict}),it)
            summary_writer.add_scalar('pnopt_pattern/loss',fxtv,it)
            summary_writer.add_scalar('pnopt_pattern/relloss',rel_loss,it)
            summary_writer.add_scalar('pnopt_pattern/relerr',rel_err,it)
            
            # if evaluate_hat_dict_for_summary_writer is not None:
            #     summary_writer.add_scalars('evaluate_hat_dict',evaluate_hat_dict_for_summary_writer(param_dict),it)
        if rel_err<err_relP or rel_loss<err_relL:
            status = 'converged: err_relP or err_relL reached'
            break
        if any([v.isnan().any() for v in x.values()]):
            status = 'nan encountered in x'
            logger.warning(f"pnopt_pattern {status} after {it} iterations")
            break
    time_elapsed = time.process_time() - time_init
    if summary_writer is not None:
        summary_writer.add_text('convergence', status, global_step=it)
        summary_writer.add_scalar('time', time_elapsed,global_step=it)
    logger.info(f"pnopt_pattern converged after {it} iterations with status {status}")
    return uncat_params(x)
