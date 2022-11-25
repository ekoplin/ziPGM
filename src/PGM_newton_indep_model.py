#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:55:01 2022

	Sufficient dimension reduction for zero-inflated graphical models
	Eric Koplin, Liliana Forzani,  Diego Tomassi, Ruth M. Pfeiffer
	
	Optimization methods for independent PGM models

@author: eric
"""
import torch
import torch.optim as optim

def PGM_newton_indep_model(*,model,X,Y,indep_params,step=1e-3,damping=1e-6,max_iter=10000,rtol=1e-4,atol=1e-9,update_hessian_each=1,fisher=False,summary_writer=None):
    assert all([k in ('eta','ThetaDiag') for k in indep_params.keys()])
    def ell(**indep_params):
        return model.ell(X,Y,indep_params)
    p             = X.size(0)
    exp_suff_stat = torch.cat([X.mean(1,keepdim=True),X.pow(2).div(2).mean(1,keepdim=True)],1).unsqueeze(2)/p
    nerr_         = torch.tensor(torch.finfo(torch.get_default_dtype()).max)
    for i in range(max_iter):
        # grad_loss_vals=torch.autograd.functional.jacobian(closure,tuple(indep_params.values()))
        # grad_loss=dict(zip(indep_params.keys(),grad_loss_vals))
        # cat_grads=torch.stack([grad_loss[k] for k in ('eta','xi','ThetaDiag') if k in indep_params.keys()],1)
        cat_grads=model.dA(X, Y, param_dict=indep_params).transpose(-1,-2).squeeze(1)-exp_suff_stat
        if not i%update_hessian_each:
            if fisher:
                Xs = model.sample(Y,indep_params)
            else:
                Xs=X
            ddA = model.ddA(X=Xs,Y=Y,param_dict=indep_params).squeeze(1)# px1x3x3
            ddA_diag=torch.diagonal(ddA,dim1=-1,dim2=-2)
            ddA_diag+=torch.ones_like(ddA_diag)*damping
        
        if not 'ThetaDiag' in indep_params.keys():
            ddA=ddA[:,:1,:1]# pxnx2x2
            cat_grads=cat_grads[:,:1,:1]
        cat_ng = torch.matmul(torch.linalg.inv(ddA),cat_grads).squeeze(-1)
        ng     = dict(zip(indep_params.keys(),torch.split(cat_ng,1,dim=1)))

        #update
        indep_params={k:v.sub_(ng[k],alpha=step) for k,v in indep_params.items()}
        
        # check convergence
        if not i%10:
            nerr=(cat_grads.squeeze(-1)*cat_ng).sum()
            if torch.isclose(nerr,nerr_,rtol,atol):
                break
            else:
                nerr_=nerr
        
            if summary_writer is not None:
                objective=ell(**indep_params)
                summary_writer.add_scalar('indep/loss',objective,i)
                summary_writer.add_scalar('indep/err',nerr,i)
    return indep_params
  
def PGM_newton_indep_regression_model(*,model,X,Y,indep_params,step=1e-3,damping=1e-6,max_iter=10000,rtol=1e-4,atol=1e-9,update_hessian_each=1,fisher=False,summary_writer=None):
    # this takes all the parameters but do not apply restrictions, acting as they were independentendent models
    assert all([k in ('eta','ThetaDiag','Gamma') for k in indep_params.keys()])
    assert 'Gamma' in indep_params.keys(), 'Gamma must be provided'
    def ell(**indep_params):
        return model.ell(X,Y,indep_params)
    p,n           = X.size()
    r             = Y.size(0)

    exp_suff_stat = torch.stack([X,X.pow(2).div(2)],2)/p# pxnx3
    nerr_         = torch.tensor(torch.finfo(torch.get_default_dtype()).max)
    for i in range(max_iter):
        # grad_loss_vals=torch.autograd.functional.jacobian(closure,tuple(indep_params.values()))
        # grad_loss=dict(zip(indep_params.keys(),grad_loss_vals))
        # dell=torch.stack([grad_loss[k] for k in ('eta','xi','ThetaDiag') if k in indep_params.keys()],1)
        dell=model.dA(X, Y, param_dict=indep_params).unsqueeze(-1)-exp_suff_stat.unsqueeze(-1)# pxnx3x1
        if not i%update_hessian_each:
            if fisher:
                Xs = model.sample(Y,indep_params)
            else:
                Xs=X
            ddA = model.ddA(X=Xs,Y=Y,param_dict=indep_params).squeeze(1)# pxnx3x3
            # compute regression part hessian
            ddA_eta=ddA[:,:,1,1]# pxnx2x2
            Hreg=torch.mean(ddA_eta.t().unsqueeze(-1).unsqueeze(-1)*torch.matmul(Y.t().unsqueeze(-1),Y.t().unsqueeze(-2)).unsqueeze(1),0)
            # compute linear-regression part hessian
            Hlinreg=torch.matmul(ddA_eta,Y.t()).unsqueeze(-2)/n#px2rx2r
            # compute lin part hessian
            if not 'ThetaDiag' in indep_params.keys():
                Hlin = ddA_eta.mean(dim=1,keepdim=True).unsqueeze(-1)# px2x2
            else:# add X^2/2 suff stat into lin part
                Hlin = ddA.mean(dim=1)# px3x3
                Hlinreg =torch.cat((Hlinreg,torch.zeros(p,1,r)),1)
            # ensemble hessian
            HH = torch.cat((torch.cat((Hlin,Hlinreg),2),torch.cat((Hlinreg.transpose(-2,-1),Hreg),2)),1)# px2+2rx2+2r 
        
        # jacobian
        if not 'ThetaDiag' in indep_params.keys():
            dell = dell[:,:,:1,:]
            db   = dell
        else:# add X^2/2 suff stat into lin part
            db   = dell[:,:,:1,:]
        Jreg=torch.matmul(db[:,:,0,0],Y.t()).unsqueeze(-1)/n#px2rx1
        Jb  =torch.mean(dell,1)
        J   = torch.cat((Jb,Jreg),1)
        # add damping
        H_diag=torch.diagonal(HH,dim1=-1,dim2=-2)
        H_diag+=torch.ones_like(H_diag)*damping

        # cat_ng = torch.matmul(torch.linalg.inv(HH),J).squeeze(-1)
        cat_ng = J.squeeze()
        ng     = dict(zip(indep_params.keys(),torch.split(cat_ng,1,dim=1)))

        #update
        indep_params={k:v.sub_(ng[k],alpha=step) for k,v in indep_params.items()}
        
        # check convergence
        if not i%100:
            nerr=(J.squeeze(-1)*cat_ng).sum()
            if torch.isclose(nerr,nerr_,rtol,atol):
                break
            else:
                nerr_=nerr
        
            if summary_writer is not None:
                objective=ell(**indep_params)
                summary_writer.add_scalar('indep/loss',objective,i)
                summary_writer.add_scalar('indep/err',nerr,i)
    return indep_params
