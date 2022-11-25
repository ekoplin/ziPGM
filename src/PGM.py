#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:25:55 2021
        
        Sufficient dimension reduction for zero-inflated graphical models
	Eric Koplin, Liliana Forzani,  Diego Tomassi, Ruth M. Pfeiffer

	Define the abstract class of pGM (PGM) models and the Normal-pGM (NPGM), 
	the Poisson-pGM (PPGM) and the TPoisson-pGM (TPPGM)
        
@author: eric
"""
import copy
import torch
from torch.autograd.functional import hessian,jacobian

from sample_truncated_poisson import sample_tpoisson
from sample_double_truncated_poisson import sample_Tpoisson

from math import pi
from scipy.stats import chi2
class PGM(object):
    def dA(self,X,Y,param_dict):
        COND=self._conditionals(X, Y, param_dict)
        if 'ThetaDiag' in param_dict.keys():
            dA = self._dA(COND['eta'],COND['theta'])#pxnx3
        else:
            dA = self._dA(COND['eta'])#pxnx3
        return dA/X.size(0)# divided by p
    def ddA(self,X,Y,param_dict):
        COND=self._conditionals(X, Y, param_dict)
        if 'ThetaDiag' in param_dict.keys():
            ddA = self._ddA(COND['eta'],COND['theta'])#pxnx3x3
        else:
            ddA = self._ddA(COND['eta'])#pxnx3x3
        return ddA/X.size(0)# divided by p
    def _cat_hessian_unnormalized(self,X,Y,param_dict,symmetric=True):
        p,n=X.size()
        r  =Y.size(0)

        ddA = self.ddA(X=X,Y=Y,param_dict=param_dict)# px1x3x3
        ddA_eta=ddA[:,:,0,0]# px1x2x2

        DjWjl=torch.matmul(X.pow(2),ddA_eta.t())
        
        if symmetric:
            DlWjl=torch.matmul(ddA_eta,X.pow(2).t())
            Hint=DjWjl+DlWjl#.transpose(0,1)# corrected 20/01/2022
        else:
            Hint=DjWjl
        Hint=Hint.unsqueeze(-1).unsqueeze(-1)
        # Hreg=torch.sum(torch.matmul(torch.matmul(Ykron.transpose(-2,-1),ddA_eta_xi),Ykron),1)#px2rx2r
        Hreg=torch.sum(ddA_eta.t().unsqueeze(-1).unsqueeze(-1)*torch.matmul(Y.t().unsqueeze(-1),Y.t().unsqueeze(-2)).unsqueeze(1),0)
        if not 'ThetaDiag' in param_dict.keys():
            Hlin = ddA_eta.sum(dim=1,keepdim=True).unsqueeze(-1)# px2x2
        else:# add X^2/2 suff stat into lin part
            Hlin = ddA.sum(dim=1)# px3x3
        return dict(inter=Hint,reg=Hreg,lin=Hlin)
       
    def cat_hessian(self,XX,YY,param_dict,symmetric=True):
        p,n=XX.size()
        r  =YY.size(0)
        if self.batch_size_hessian <= 0: # use full batch
            batch_size = n
        else:
            batch_size = self.batch_size_hessian

        Hlin = torch.zeros(p,2 if 'ThetaDiag' in param_dict.keys() else 1,2 if 'ThetaDiag' in param_dict.keys() else 1)
        Hreg = torch.zeros(p,r,r)
        Hint = torch.zeros(p,p,1,1)
        
        for X,Y in zip(torch.split(XX,batch_size,1),torch.split(YY,batch_size,1)):
            nHdict = self._cat_hessian_unnormalized(X,Y,param_dict,symmetric)
            Hint  += nHdict['inter']
            Hreg  += nHdict['reg']
            Hlin  += nHdict['lin']
        
        Hint/=n
        Hreg/=n
        Hlin/=n
        
        # set diagonals to the identity
        Hint[range(p),range(p),:,:]=1
        return dict(inter=Hint,reg=Hreg,lin=Hlin)

    def _cat_jacobian_unnormalized(self,X, Y,param_dict,symmetric=True):
        p,n=X.size()
        r  =Y.size(0)
        # dl=[-Exj dA/detaj, -Enuj dA/dxi -Ex^2/2 dA/dtheta]
        ET=torch.stack((X,X.pow(2)/2),-1)/p
        
        dA = self.dA(X=X,Y=Y,param_dict=param_dict)-ET# pxnx1
        dA_eta=dA[:,:,0]# pxnx1
        
        DjWjl=torch.matmul(X,dA_eta.t())#px1 
        if symmetric:
            DlWjl=torch.matmul(dA_eta,X.t())#px1
            Jint=DjWjl+DlWjl
        else:
            Jint=DjWjl
        Jint=Jint.unsqueeze(-1).unsqueeze(-1)
        # add zeros along the diagonals 
        Jint[range(p),range(p),:,:]=0.
        
        Jreg=torch.matmul(dA_eta,Y.t()).unsqueeze(-1)#px2rx1
        if not 'ThetaDiag' in param_dict.keys():
            Jlin = dA_eta.sum(dim=1).unsqueeze(-1).unsqueeze(-1)# px2x1
        else:# add X^2/2 suff stat into lin part
            Jlin = dA.sum(dim=1).unsqueeze(-1)# px3x1
        return dict(inter=Jint,reg=Jreg,lin=Jlin)
        
    def cat_jacobian(self,XX,YY,param_dict,symmetric=True):
        p,n=XX.size()
        r  =YY.size(0)
        if self.batch_size_jacobian <= 0: # use full batch
            batch_size = n
        else:
            batch_size = self.batch_size_jacobian

        Jlin = torch.zeros(p,2 if 'ThetaDiag' in param_dict.keys() else 1,1)
        Jreg = torch.zeros(p,r,1)
        Jint = torch.zeros(p,p,1,1)
        
        for X,Y in zip(torch.split(XX,batch_size,1),torch.split(YY,batch_size,1)):
            nJdict = self._cat_jacobian_unnormalized(X,Y,param_dict,symmetric)
            Jint  += nJdict['inter']
            Jreg  += nJdict['reg']
            Jlin  += nJdict['lin']
        
        Jint/=n
        Jreg/=n
        Jlin/=n
        
        return dict(inter=Jint,reg=Jreg,lin=Jlin)
    def __init__(self,reduced_rank=False,d=1,dG=1,dP=1,batch_size_jacobian=0,batch_size_hessian=0):
        # if reduced rank, Gamma and Psi are projected in project_regression method
        # if joint, we consider rank(Gamma;Psi) <=d , otherwise, rank(Gamma)<=dG and rank(Psi)<=dP
        self.batch_size_jacobian = batch_size_jacobian
        self.batch_size_hessian  = batch_size_hessian
        # self.joint = joint
        if reduced_rank:
           self.project_regression = lambda Gamma: self.project_Gamma_Psi_redRank(Gamma=Gamma,d=d)
        else:
             self.project_regression=self.project_Gamma_Psi_identity
    @classmethod
    def _Ano0(cls,eta_cond,**kwargs):
        pass
    @classmethod
    def _conditionals(cls,X,Y,param_dict):
        pass
    @classmethod
    def ell(cls,X,Y,param_dict):
        pass
    #@classmethod, will depend on each instance
    def project_Theta(self,Theta,ThetaDiag=None):
        # allows for restrictions on Theta matrix, it sould be passed to optimization routine
        # it could affect ThetaDiag, but is not desiderable. it is here only inc case we need to normalize Theta as an (inverse) correlation. 
        # must return a zero diagonal matrix
        return Theta,ThetaDiag
    #@classmethod, will depend on each instance
    def project_interacts(self,*,Theta,ThetaDiag=None):
        Theta,ThetaDiag=self.project_Theta((Theta+Theta.t())/2,ThetaDiag=ThetaDiag)# depends on each model
        # Phi=(Phi+PhiT.t())/2
        # PhiT=Phi.t().clone()#@todo: see if clone is needed
        # Lambda=(Lambda+Lambda.t())/2
        return {**dict(Theta=Theta),**(dict(ThetaDiag=ThetaDiag) if ThetaDiag is not None else {})}
    @classmethod
    def project_Gamma_Psi_identity(cls,*,Gamma):
        # allows for restrictions on regression matrices, it sould be passed to optimization routine
        return dict(Gamma=Gamma)
    @classmethod
    def project_Gamma_Psi_redRank(cls,*,Gamma,d=1):
        # allows for restrictions on regression matrices, it sould be passed to optimization routine
        GP = Gamma
        U, S, Vh = torch.linalg.svd(GP, full_matrices=False)
        # Return rescaled singular vectors
        S[d:]=0
        Gamma=U @ torch.diag_embed(S) @ Vh
        # Gamma,Psi=torch.split(GP,Gamma.size(0),0)
        return dict(Gamma=Gamma)
    @classmethod
    def _conditional_sample(cls,COND):
        pass
    # @classmethod
    def sample(self,Y,param_dict,burn_in=int(1e4),X0=None,H0=None):
        n=Y.size(1)
        p=param_dict['eta'].size(0)
        if X0 is None or H0 is None:
            X0=torch.zeros(p,n)
            H0=torch.zeros(p,n)
        COND= self._conditionals(X0,Y,param_dict) 
        if ('Theta' in param_dict.keys()) and not (param_dict['Theta'].norm()==0):
            for _ in range(int(burn_in)):
                X=self._conditional_sample(COND)
                COND=self._conditionals(X,Y,param_dict)
        if not any([k in param_dict.keys() for k in ('Gamma','Theta')]):# indep case
            for k,v in COND.items():
                COND[k]=v.expand(-1,n)
        
        X=self._conditional_sample(COND)
        return X
    def sample_count(self,Y,param_dict,pattern_dict=None,burn_in=int(1e4),TXrefMean=100,separate_sampling=True):
        # TXrefMean is used to generate a reference used in inverse log transform.
        # if separate sampling, the indep variables are sampled without burn-in 
        p = param_dict['eta'].size(0)
        if separate_sampling and pattern_dict is not None:
            # separate into independent and inmteractions model
            vars_with_interaction = torch.bitwise_or(pattern_dict['ci'].any(0),pattern_dict['ci'].any(1))#.reshape(args['p'],1)

            param_inter         = {k:param_dict[k][vars_with_interaction] for k in ['eta','xi','Gamma','Psi','ThetaDiag'] if k in param_dict.keys() }
            param_inter.update({k:param_dict[k][vars_with_interaction,:][:,vars_with_interaction] for k in ['Theta','Phi','Lambda']})
            param_inter['PhiT'] = param_inter['Phi'].t()
            param_indep         = {k:param_dict[k][~vars_with_interaction] for k in ['eta','xi','Gamma','Psi','ThetaDiag'] if k in param_dict.keys() }
            param_indep.update({k:param_dict[k][~vars_with_interaction,:][:,~vars_with_interaction] for k in ['Theta','Phi','Lambda']})
            param_indep['PhiT'] =param_indep['Phi'].t()
        else:
            param_inter = param_dict
            param_inter['PhiT'] = param_inter['Phi'].t()
            
            param_indep = None
            
        # sample dependent model
        ZZ_inter = self.sample(Y, param_inter,burn_in=10000)
        # sample independent model
        if param_indep is not None:
            ZZ_indep = self.sample(Y, param_indep,burn_in=0)
            # ensemble
            ZZ                           = torch.zeros(p,Y.size(1))
            ZZ[vars_with_interaction,:]  = ZZ_inter
            ZZ[~vars_with_interaction,:] = ZZ_indep
            # HH                           = torch.zeros(p,Y.size(1))
            # HH[vars_with_interaction,:]  = HH_inter
            # HH[~vars_with_interaction,:] = HH_indep
        else:
            ZZ = ZZ_inter
            # HH = HH_inter
        # if normal convert to counts
        XXk=sample_tpoisson(TXrefMean*torch.ones(1,Y.size(1)))
        if self.__class__.__name__=='HN':
            XX=(XXk*ZZ.exp()*HH).ceil()
        else:
            XX=ZZ
        return XX,XXk 
        
    @classmethod
    def transform_data(cls,X,Y,y=None,**kwargs):
        return X,H,Y,y
    @classmethod
    def init_dict(cls,X,Y):
        # return indep model params
        pass
    @classmethod
    def dim_count(cls,param_dict,pattern_dict):
        p,r=param_dict['Gamma'].shape
        selected_params =pattern_dict['vs'].sum()*r+pattern_dict['ci'].triu(1).sum()# free parameters
        return selected_params+p+(p if 'ThetaDiag' in param_dict.keys() else 0)
    @classmethod
    def AIC(cls,ell,dim,p,n):
        return 2*ell*p*n+2*dim
    @classmethod
    def BIC(cls,ell,dim,p,n):
        return 2*ell*p*n+dim*torch.tensor(n).log()
    def compute_AIC_BIC(self,X,Y,param_dict,pattern_dict,count_dim=False):
        p,n=X.size()
        ell_pattern=self.ell(X,Y,{k:v if k in ('eta','ThetaDiag') else (torch.where(pattern_dict['vs'].reshape(p,1),v,torch.zeros(1)) if k in ['Gamma'] else torch.where(pattern_dict['ci'],v,torch.zeros(1))) for k,v in param_dict.items()})
        if not count_dim:
            raise NotImplementedError()
        else:
            dim=self.dim_count(param_dict,pattern_dict)
            returns = {}
        AIC=self.AIC(ell_pattern,dim,p,n)
        BIC=self.BIC(ell_pattern,dim,p,n)
        return AIC,BIC,{'ell':-ell_pattern*p*n,'dim':dim,**returns}

class BPGM(PGM):
    @classmethod
    def _A(self,eta):
        return (1+eta.exp()).log()
    @classmethod
    def _dA(cls,eta,theta=None):
        dA=torch.sigmoid(eta)
        return torch.stack((dA,torch.zeros_like(eta)),-1)# pxnx2 eta,thetadiag
    @classmethod
    def _ddA(cls,eta,theta=None):
        dA  = cls._dA(eta)[:,:,0]
        ddA = dA*(1-dA)
        return  torch.stack((
                torch.stack((ddA                  , torch.zeros_like(eta)),-1),
                torch.stack((torch.zeros_like(eta), torch.zeros_like(eta)),-1)
                ),-1)
    @classmethod
    def _conditionals(cls,X,Y,param_dict):
        if set(['Gamma','Theta']).issubset(set(param_dict.keys())):
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y) + torch.mm(param_dict['Theta'],X)# here Theta is assumed to be zero diagonal!
            #theta=param_dict['Theta'].diag()
        elif set(['Gamma']).issubset(set(param_dict.keys())) and not any([k in ('Theta',) for k in param_dict.keys()]):# indep with regression
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y)
        elif not any([k in ('Gamma','Theta') for k in param_dict.keys()]):# indept model
            eta_cond=param_dict['eta']
        else:
            raise AssertionError('invalid set of parameters in param_dict')
        A=cls._A(eta_cond)
        return dict(eta=eta_cond,A=A)
    @classmethod
    def ell(cls,X,Y,param_dict):
        COND = cls._conditionals(X,Y,param_dict)
        S=-(COND['eta']*X-COND['A'])
        return S[~S.isinf()].mean()
    @classmethod
    def _conditional_sample(cls,COND):
        H=torch.distributions.bernoulli.Bernoulli(torch.sigmoid(COND['eta'])).sample()
        return H
    def transform_data(self,X,Y,y=None):
        return X.bool().to(torch.get_default_dtype()),Y,y
    @classmethod
    def init_dict(cls,X,Y):
        # X must be the transformed variable!        
        mean=torch.clamp(torch.mean(X,1,keepdims=True),min=1e-3)
        eta=torch.logit(mean)
        return dict(eta=eta)            
        
class PPGM(PGM):
    @classmethod
    def _A(self,eta):
        return eta.exp()
    @classmethod
    def _dA(cls,eta,theta=None):
        mu=eta.exp()
        dA=mu
        return torch.stack((dA,torch.zeros_like(eta)),-1)# pxnx2 eta,thetadiag
    @classmethod
    def _ddA(cls,eta,theta=None):
        mu=eta.exp()
        ddA=mu
        return  torch.stack((
                torch.stack((ddA                  , torch.zeros_like(eta)),-1),
                torch.stack((torch.zeros_like(eta), torch.zeros_like(eta)),-1)
                ),-1)
    @classmethod
    def project_Theta(cls,Theta,ThetaDiag=None):
        return  torch.clamp(Theta,max=0),None
    
    @classmethod
    def _conditionals(cls,X,Y,param_dict):
        if set(['Gamma','Theta']).issubset(set(param_dict.keys())):
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y) + torch.mm(param_dict['Theta'],X)# here Theta is assumed to be zero diagonal!
            #theta=param_dict['Theta'].diag()
        elif set(['Gamma']).issubset(set(param_dict.keys())) and not any([k in ('Theta',) for k in param_dict.keys()]):# indep with regression
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y)
        elif not any([k in ('Gamma','Theta') for k in param_dict.keys()]):# indept model
            eta_cond=param_dict['eta']
        else:
            raise AssertionError('invalid set of parameters in param_dict')
        A=cls._A(eta_cond)
        return dict(eta=eta_cond,A=A)
    @classmethod
    def ell(cls,X,Y,param_dict):
        COND = cls._conditionals(X,Y,param_dict)
        S=-(COND['eta']*X-torch.lgamma(X+1)-COND['A'])
        return S[~S.isinf()].mean()
    @classmethod
    def _conditional_sample(cls,COND):
        X=torch.poisson(COND['eta'].exp())
        return X
    @classmethod
    def init_dict(cls,X,Y):
        # X must be the transformed variable!        
        mean=torch.clamp(torch.mean(X,1,keepdims=True),min=1e-3)
        eta=mean.log()
        return dict(eta=eta)

class TPPGM(PGM):
    def _A(self,eta):
        return eta.exp()+torch.log(torch.igammac(self.Tstar+1,eta.exp()))
    def _dA(self,eta,theta=None):
        mu=eta.exp()
        dA = mu*torch.igammac(self.Tstar,mu)/torch.igammac(self.Tstar+1,mu)       
        return torch.stack((dA,torch.zeros_like(eta)),-1)# pxnx2 eta,thetadiag
    def _ddA(self,eta,theta=None):
        mu=eta.exp()
        dA  = mu*torch.igammac(self.Tstar,mu)/torch.igammac(self.Tstar+1,mu) 
        
        PTstar  =(-mu+self.Tstar*eta-torch.lgamma(self.Tstar+1)).exp()#/factorial(self.Tstar)
        PTstarm1=(-mu+(self.Tstar-1)*eta-torch.lgamma(self.Tstar)).exp()
        PTstarp1=(-mu+(self.Tstar+1)*eta-torch.lgamma(self.Tstar+2)).exp()
        PlTstarp1=torch.igammac(self.Tstar+1,mu)
        
        ddA = dA+mu.pow(2)*((PTstar-PTstarm1)/PlTstarp1-(PTstar*PTstarp1)/PlTstarp1.pow(2))
       
        return  torch.stack((
                torch.stack((ddA                  ,torch.zeros_like(eta), torch.zeros_like(eta)),-1),
                torch.stack((torch.zeros_like(eta),torch.zeros_like(eta), torch.zeros_like(eta)),-1)
                ),-1)
    
    def __init__(self,Tstar=1e4,**kwargs):
        super(TPPGM,self).__init__(**kwargs)
        self.Tstar=torch.tensor(Tstar)
    def _conditionals(self,X,Y,param_dict):
        if set(['Gamma','Theta']).issubset(set(param_dict.keys())):
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y) + torch.mm(param_dict['Theta'],X)# here Theta is assumed to be zero diagonal!
            #theta=param_dict['Theta'].diag()
        elif set(['Gamma']).issubset(set(param_dict.keys())) and not any([k in ('Theta',) for k in param_dict.keys()]):# indep with regression
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y)
        elif not any([k in ('Gamma','Theta') for k in param_dict.keys()]):# indept model
            eta_cond=param_dict['eta']
        else:
            raise AssertionError('invalid set of parameters in param_dict')

        A=self._A(eta_cond)
        return dict(eta=eta_cond,A=A)
    def ell(self,X,Y,param_dict):
        COND = self._conditionals(X,Y,param_dict)
        S=-(COND['eta']*X-torch.lgamma(X+1)-COND['A'])
        return S[~S.isinf()].mean()
    def _conditional_sample(self,COND):
        X=sample_Tpoisson(COND['eta'].exp(),self.Tstar)
        return X
    def transform_data(self,X,Y,y=None):
        return X.clamp_max(self.Tstar),Y,y
    def init_dict(self,X,Y):
        # X must be the transformed variable!
        mean=torch.clamp(torch.mean(X,1,keepdims=True),min=1e-3)
        eta=mean.log()
        return dict(eta=eta)

class NPGM(PGM):
    @classmethod
    def _A(cls,eta,theta):
        return -eta.pow(2)/2/theta-torch.log(-theta/2/pi)/2# corrected 21/10/25
    @classmethod
    def _dA(cls,eta,theta):
        return torch.stack((-eta/theta,eta.pow(2)/theta.pow(2)/2-1/theta/2),-1)# pxnx2
    @classmethod
    def _ddA(cls,eta,theta):
        return  torch.stack((
                torch.stack((-1/theta.expand_as(eta), eta/theta.pow(2)),-1),
                torch.stack((eta/theta.pow(2)       ,-eta.pow(2)/theta.pow(3)+1/theta.pow(2)/2),-1)
                ),-1)
    
    # pxnx2x2
    def __init__(self,project_theta_onto_negDef=False,**kwargs):
        super(NPGM,self).__init__(**kwargs)
        if project_theta_onto_negDef:
            self.project_Theta=self.project_onto_negDef
        
    def project_onto_negDef(cls,Theta,ThetaDiag):# projects onto semidef pos
        A=Theta+torch.diag_embed(ThetaDiag.squeeze())
        L, Q = torch.linalg.eigh((A+A.t())/2)
        QLQ  = Q @ torch.diag_embed(L.clamp_max(0)) @ Q.transpose(-2, -1)
        # separate
        ThetaDiag =QLQ.diag().unsqueeze(1)
        Theta     =QLQ.fill_diagonal_(0)
        return Theta,ThetaDiag

    @classmethod
    def _conditionals(cls,X,Y,param_dict):
        if set(['Gamma','Theta']).issubset(set(param_dict.keys())):
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y) + torch.mm(param_dict['Theta'],X)# here Theta is assumed to be zero diagonal!
        elif set(['Gamma']).issubset(set(param_dict.keys())) and not any([k in ('Theta',) for k in param_dict.keys()]):# indep with regression
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y)
        elif not any([k in ('Gamma','Theta') for k in param_dict.keys()]):# indept model
            eta_cond=param_dict['eta']
        else:
            raise AssertionError('invalid set of parameters in param_dict')
        theta=param_dict['ThetaDiag'].reshape(-1,1)

        A=cls._A(eta_cond, theta)
        
        return dict(eta=eta_cond,theta=theta,A=A)
    @classmethod
    def ell(cls,X,Y,param_dict):
        COND = cls._conditionals(X,Y,param_dict)
        S=-(COND['eta']*X+COND['theta']*X.pow(2)/2-COND['A'])
        return S[~S.isinf()].mean()#/p/n
    @classmethod
    def _conditional_sample(cls,COND):       
        sigma=(-1/COND['theta']).sqrt().expand_as(COND['eta'])
        mu=-COND['eta']/COND['theta']
        X=mu+sigma*torch.randn_like(COND['eta'])
        return X
    @classmethod
    def find_k(H):
        assert X.bool().all(1).any(),'H must have a variable that is always 1, remove samples'
        return int(torch.where(X.bool().all(1))[0])
    @classmethod
    def transform_data(cls,X,Y,y=None):# agree with sample_count
        Z  = torch.log(X/X.sum(0,keepdims=True))
        Z[~Z.bool()]=torch.randn((~Z.bool()).sum())*1e-3
        Z[~H.bool()]=0
        assert (Z.bool()==X.bool()).all(),'zeros change'
        # Hnok=Znok.bool().to(X.dtype)
        return Z,Y,y

    @classmethod
    def init_dict(cls,X,Y):
        # X must be the transformed variable!
        mean=torch.mean(X,1,keepdims=True)
        var =torch.clamp(torch.var(X,1,keepdims=True),min=1e-3)
        eta=mean/var
        ThetaDiag=-1/var
        return dict(eta=eta,ThetaDiag=ThetaDiag)
