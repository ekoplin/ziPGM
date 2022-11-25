#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:25:55 2021
	
	Sufficient dimension reduction for zero-inflated graphical models
	Eric Koplin, Liliana Forzani,  Diego Tomassi, Ruth M. Pfeiffer

	Define the abstract class of zi-pGM (HM) models and the Normal-zipGM (HN), 
	the Poisson-zipGM (HP) and the TPoisson-zipGM (HTP)
	
@author: eric
"""
import copy
import torch
from torch.autograd.functional import hessian,jacobian
from math import factorial

from sample_truncated_poisson import sample_tpoisson
from sample_double_truncated_poisson import sample_tTpoisson

from math import pi
from scipy.stats import chi2
class HM(object):
    def dA(self,X,H,Y,param_dict):
        COND=self._conditionals(X, H, Y, param_dict)
        if 'ThetaDiag' in param_dict.keys():
            dAn0 = self._dAn0(COND['eta'],COND['theta'])#pxnx3
        else:
            dAn0 = self._dAn0(COND['eta'])#pxnx3
        s=torch.sigmoid(COND['xi']+COND['Ano0']).unsqueeze(-1)# pxnx1
        p,n=COND['eta'].size()
        dA=dAn0
        dA[:,:,1]=1
        
        return s*dA/X.size(0)# divided by p
    def ddA(self,X,H,Y,param_dict):
        COND=self._conditionals(X, H, Y, param_dict)
        if 'ThetaDiag' in param_dict.keys():
            dAn0 = self._dAn0(COND['eta'],COND['theta'])#pxnx3
            ddAn0 = self._ddAn0(COND['eta'],COND['theta'])#pxnx3x3
        else:
            dAn0 = self._dAn0(COND['eta'])#pxnx3
            ddAn0 = self._ddAn0(COND['eta'])#pxnx3x3
        s=torch.sigmoid(COND['xi']+COND['Ano0']).unsqueeze(-1).unsqueeze(-1)# pxnx1x1
        p,n=COND['eta'].size()
        # dA =torch.cat((torch.ones(p,n,1),dAn0),-1)# pxnx3
        dA=dAn0
        dA[:,:,1]=1
        
        ddA=ddAn0
        return (s*(1-s)*torch.einsum('abi,abj->abij',(dA,dA))+s*ddA)/X.size(0)# divided by p
    
    def _cat_hessian_unnormalized(self,X,H,Y,param_dict,symmetric=True):
        p,n=X.size()
        r  =Y.size(0)

        ddA = self.ddA(X=X,H=H,Y=Y,param_dict=param_dict)# px1x3x3
        ddA_eta_xi=ddA[:,:,:2,:2]# px1x2x2
        
        XH    =torch.stack((X,H),-1)#pxnx2
     
        XHkron=torch.kron(torch.eye(2).unsqueeze(0).unsqueeze(0),XH.unsqueeze(-2))#pxnx2x4    WARNING MEMORY INTENSIVE
        kronXH=torch.kron(XH.unsqueeze(-2),torch.eye(2).unsqueeze(0).unsqueeze(0))#pxnx2x4    WARNING MEMORY INTENSIVE
        Ykron=torch.kron(torch.eye(2).unsqueeze(0).unsqueeze(0),Y.expand(p,r,n).permute(0,2,1).unsqueeze(-2).contiguous())#pxnx2x2r    WARNING MEMORY INTENSIVE

        DjWjl=torch.sum(torch.matmul(torch.matmul(XHkron.unsqueeze(0).transpose(-2,-1),ddA_eta_xi.unsqueeze(1)),XHkron.unsqueeze(0)),2)#px4x4 # corrected 20/01/2022
        if symmetric:
            DlWjl=torch.sum(torch.matmul(torch.matmul(kronXH.unsqueeze(0).transpose(-2,-1),ddA_eta_xi.unsqueeze(1)),kronXH.unsqueeze(0)),2)#px4x4 # corrected 20/01/2022
            Hint=DjWjl+DlWjl.transpose(0,1)# corrected 20/01/2022
        else:
            Hint=DjWjl
        
        Hreg=torch.sum(torch.matmul(torch.matmul(Ykron.transpose(-2,-1),ddA_eta_xi),Ykron),1)#px2rx2r
        if not 'ThetaDiag' in param_dict.keys():
            Hlin = ddA_eta_xi.sum(dim=1)# px2x2
        else:# add X^2/2 suff stat into lin part
            Hlin = ddA.sum(dim=1)# px3x3
        return dict(inter=Hint,reg=Hreg,lin=Hlin)
       
    def cat_hessian(self,XX,HH,YY,param_dict,symmetric=True):
        p,n=XX.size()
        r  =YY.size(0)
        if self.batch_size_hessian <= 0: # use full batch
            batch_size = n
        else:
            batch_size = self.batch_size_hessian

        Hlin = torch.zeros(p,3 if 'ThetaDiag' in param_dict.keys() else 2,3 if 'ThetaDiag' in param_dict.keys() else 2)
        Hreg = torch.zeros(p,2*r,2*r)
        Hint = torch.zeros(p,p,4,4)
        
        for X,H,Y in zip(torch.split(XX,batch_size,1),torch.split(HH,batch_size,1),torch.split(YY,batch_size,1)):
            nHdict = self._cat_hessian_unnormalized(X,H,Y,param_dict,symmetric)
            Hint  += nHdict['inter']
            Hreg  += nHdict['reg']
            Hlin  += nHdict['lin']
        
        Hint/=n
        Hreg/=n
        Hlin/=n
        
        # set diagonals to the identity
        # Hint+=regularization*torch.eye(4).reshape(1,1,4,4)# induce positive definiteness
        Hint[range(p),range(p),:,:]=torch.eye(4)
        return dict(inter=Hint,reg=Hreg,lin=Hlin)

    def _cat_jacobian_unnormalized(self,X,H,Y,param_dict,symmetric=True):
        p,n=X.size()
        r  =Y.size(0)
        # dl=[-Exj dA/detaj, -Enuj dA/dxi -Ex^2/2 dA/dtheta]
        ET=torch.stack((X,H,X.pow(2)/2),-1).unsqueeze(-1)/p
        
        dA = self.dA(X=X,H=H,Y=Y,param_dict=param_dict).unsqueeze(-1)-ET# pxnx3x1
        dA_eta_xi=dA[:,:,:2,:]# pxnx2x1
        
        XH    =torch.stack((X,H),-1)#pxnx2
        
        XHkron=torch.kron(torch.eye(2).unsqueeze(0).unsqueeze(0),XH.unsqueeze(-2))#pxnx2x4    WARNING MEMORY INTENSIVE
        kronXH=torch.kron(XH.unsqueeze(-2),torch.eye(2).unsqueeze(0).unsqueeze(0))#pxnx2x4    WARNING MEMORY INTENSIVE
        Ykron=torch.kron(torch.eye(2).unsqueeze(0).unsqueeze(0),Y.expand(p,r,n).permute(0,2,1).unsqueeze(-2).contiguous())#pxnx2x2r    WARNING MEMORY INTENSIVE
        
        DjWjl=torch.sum(torch.matmul(XHkron.unsqueeze(0).transpose(-2,-1),dA_eta_xi.unsqueeze(1)),2)#px4x4 # corrected 20/01/2022
        if symmetric:
            DlWjl=torch.sum(torch.matmul(kronXH.unsqueeze(0).transpose(-2,-1),dA_eta_xi.unsqueeze(1)),2)#px4x4 # corrected 20/01/2022
            Jint=DjWjl+DlWjl.transpose(0,1)# corrected 20/01/2022
        else:
            Jint=DjWjl
        # add zeros along the diagonals 
        Jint[range(p),range(p),:,:]=0.
        
        Jreg=torch.sum(torch.matmul(Ykron.transpose(-2,-1),dA_eta_xi),1)#px2rx1
        if not 'ThetaDiag' in param_dict.keys():
            Jlin = dA_eta_xi.sum(dim=1)# px2x1
        else:# add X^2/2 suff stat into lin part
            Jlin = dA.sum(dim=1)# px3x1
        return dict(inter=Jint,reg=Jreg,lin=Jlin)
        
    def cat_jacobian(self,XX,HH,YY,param_dict,symmetric=True):
        p,n=XX.size()
        r  =YY.size(0)
        if self.batch_size_jacobian <= 0: # use full batch
            batch_size = n
        else:
            batch_size = self.batch_size_jacobian

        Jlin = torch.zeros(p,3 if 'ThetaDiag' in param_dict.keys() else 2,1)
        Jreg = torch.zeros(p,2*r,1)
        Jint = torch.zeros(p,p,4,1)
        
        for X,H,Y in zip(torch.split(XX,batch_size,1),torch.split(HH,batch_size,1),torch.split(YY,batch_size,1)):
            nJdict = self._cat_jacobian_unnormalized(X,H,Y,param_dict,symmetric)
            Jint  += nJdict['inter']
            Jreg  += nJdict['reg']
            Jlin  += nJdict['lin']
        
        Jint/=n
        Jreg/=n
        Jlin/=n
        
        return dict(inter=Jint,reg=Jreg,lin=Jlin)
    def __init__(self,reduced_rank=False,joint=True,d=1,dG=1,dP=1,batch_size_jacobian=0,batch_size_hessian=0):
        # if reduced rank, Gamma and Psi are projected in project_regression method
        # if joint, we consider rank(Gamma;Psi) <=d , otherwise, rank(Gamma)<=dG and rank(Psi)<=dP
        self.batch_size_jacobian = batch_size_jacobian
        self.batch_size_hessian  = batch_size_hessian
        self.joint = joint
        if reduced_rank:
            if joint:
                self.project_regression = lambda Gamma,Psi: self.project_Gamma_Psi_redRank_joint(Gamma=Gamma,Psi=Psi,d=d)
            else:
                self.project_regression = lambda Gamma,Psi: self.project_Gamma_Psi_redRank_staked(Gamma=Gamma,Psi=Psi,dG=dG,dP=dP)
        else:
             self.project_regression=self.project_Gamma_Psi_identity
    @classmethod
    def _Ano0(cls,eta_cond,**kwargs):
        pass
    @classmethod
    def _conditionals(cls,X,H,Y,param_dict):
        pass
    @classmethod
    def ell(cls,X,H,Y,param_dict):
        pass
    #@classmethod, will depend on each instance
    def project_Theta(self,Theta,ThetaDiag=None):
        # allows for restrictions on Theta matrix, it sould be passed to optimization routine
        return Theta,ThetaDiag
    #@classmethod, will depend on each instance
    def project_interacts(self,*,Theta,PhiT,Phi,Lambda,ThetaDiag=None):
        Theta,ThetaDiag=self.project_Theta((Theta+Theta.t())/2,ThetaDiag=ThetaDiag)# depends on each model
        Phi=(Phi+PhiT.t())/2
        PhiT=Phi.t().clone()#@todo: see if clone is needed
        Lambda=(Lambda+Lambda.t())/2
        return {**dict(Theta=Theta,PhiT=PhiT,Phi=Phi,Lambda=Lambda),**(dict(ThetaDiag=ThetaDiag) if ThetaDiag is not None else {})}
    @classmethod
    def project_Gamma_Psi_identity(cls,*,Gamma,Psi):
        # allows for restrictions on regression matrices, it sould be passed to optimization routine
        return dict(Gamma=Gamma,Psi=Psi)
    @classmethod
    def project_Gamma_Psi_redRank_joint(cls,*,Gamma,Psi,d=1):
        # allows for restrictions on regression matrices, it sould be passed to optimization routine
        GP = torch.cat((Gamma,Psi),0)
        U, S, Vh = torch.linalg.svd(GP, full_matrices=False)
        # Return rescaled singular vectors
        S[d:]=0
        GP=U @ torch.diag_embed(S) @ Vh
        Gamma,Psi=torch.split(GP,Gamma.size(0),0)
        return dict(Gamma=Gamma,Psi=Psi)
    @classmethod
    def project_Gamma_Psi_redRank_staked(cls,*,Gamma,Psi,dG=1,dP=1):
        # allows for restrictions on regression matrices, it sould be passed to optimization routine
        
        reduced=[]
        for GP,d in [(Gamma,dG),(Psi,dP)]:
            U, S, Vh = torch.linalg.svd(GP, full_matrices=False)
            # Return rescaled singular vectors
            S[d:]=0
            reduced.append(U @ torch.diag_embed(S) @ Vh)
        return dict(zip(['Gamma','Psi'],reduced))
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
        COND= self._conditionals(X0,H0,Y,param_dict) 
        if any([k in param_dict.keys() for k in ('Theta','PhiT','Phi','Lambda')]) and not (param_dict['Theta'].norm()==0 and param_dict['Phi'].norm()==0 and param_dict['Lambda'].norm()==0):
            for _ in range(int(burn_in)):
                #del X
                #torch.cuda.empty_cache()
                X,H=self._conditional_sample(COND)
                #X[H]=0
                #del Mu
                #torch.cuda.empty_cache()
                COND=self._conditionals(X,H,Y,param_dict)
        #del X
        #torch.cuda.empty_cache()
        if not any([k in param_dict.keys() for k in ('Gamma','Psi','Theta','PhiT','Phi','Lambda')]):# indep case
            for k,v in COND.items():
                COND[k]=v.expand(-1,n)
        
        X,H=self._conditional_sample(COND)
        #del Mu
        #torch.cuda.empty_cache()
        return X,H
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
        ZZ_inter,HH_inter = self.sample(Y, param_inter,burn_in=10000)
        # sample independent model
        if param_indep is not None:
            ZZ_indep,HH_indep = self.sample(Y, param_indep,burn_in=0)
            # ensemble
            ZZ                           = torch.zeros(p,Y.size(1))
            ZZ[vars_with_interaction,:]  = ZZ_inter
            ZZ[~vars_with_interaction,:] = ZZ_indep
            HH                           = torch.zeros(p,Y.size(1))
            HH[vars_with_interaction,:]  = HH_inter
            HH[~vars_with_interaction,:] = HH_indep
        else:
            ZZ = ZZ_inter
            HH = HH_inter
        # if normal convert to counts
        XXk=sample_tpoisson(TXrefMean*torch.ones(1,Y.size(1)))
        if self.__class__.__name__=='HN':
            XX=(XXk*ZZ.exp()*HH).ceil()
        else:
            XX=ZZ
        return XX,HH,XXk 
        
    @classmethod
    def transform_data(cls,X,H,Y,y=None,**kwargs):
        return X,H,Y,y
    @classmethod
    def init_dict(cls,X,H,Y):
        # return indep model params
        pass

    def estimate_sensitivity_matrix(self,X,H,Y,param_dict,pattern_dict):
        # http://www3.stat.sinica.edu.tw/sstest/oldpdf/A21n11.pdf#page=32&zoom=auto,-201,821
        p=X.size(0)
                
        Hdict = self.cat_hessian(X, H, Y, param_dict)
        
        blocks   =[ torch.block_diag(*[v.squeeze() for v in torch.split(Hdict['lin'],1)]),
                    torch.block_diag(*[v.squeeze() for v in torch.split(Hdict['reg'][pattern_dict['vs'].squeeze()],1)]) if pattern_dict['vs'].any() else None,
                    torch.block_diag(*[v.squeeze() for v in torch.split(Hdict['inter'][pattern_dict['ci'].triu(1)],1)]) if pattern_dict['ci'].triu(1).any() else None
                    ]

        return torch.block_diag(*[v for v in blocks if v is not None])*p# because loss is normalized
    def estimate_inv_sensitivity_matrix(self,X,H,Y,param_dict,pattern_dict):
        # http://www3.stat.sinica.edu.tw/sstest/oldpdf/A21n11.pdf#page=32&zoom=auto,-201,821
        p=X.size(0)
                
        Hdict    = self.cat_hessian(X, H, Y, param_dict)# normalized by p and n
        invHdict = {k:torch.linalg.inv(v) for k,v in Hdict.items()}
        
        blocks   =[ torch.block_diag(*[v.squeeze() for v in torch.split(invHdict['lin'],1)]),
                    torch.block_diag(*[v.squeeze() for v in torch.split(invHdict['reg'][pattern_dict['vs'].squeeze()],1)]) if pattern_dict['vs'].any() else None,
                    torch.block_diag(*[v.squeeze() for v in torch.split(invHdict['inter'][pattern_dict['ci'].triu(1)],1)]) if pattern_dict['ci'].triu(1).any() else None
                    ]

        return torch.block_diag(*[v for v in blocks if v is not None])/p# because loss is normalized
   
    def estimate_variability_matrix(self,X,H,Y,param_dict,pattern_dict):
        # http://www3.stat.sinica.edu.tw/sstest/oldpdf/A21n11.pdf#page=32&zoom=auto,-201,821
        p,n=X.size()
        r  =Y.size(0)
        # dl=[-Exj dA/detaj, -Enuj dA/dxi -Ex^2/2 dA/dtheta]
        ET=torch.stack((X,H,X.pow(2)/2),-1).unsqueeze(-1)/p
        
        dA = self.dA(X=X,H=H,Y=Y,param_dict=param_dict).unsqueeze(-1)-ET# pxnx3x1
        dA_eta_xi=dA[:,:,:2,:]# pxnx2x1
        if not 'ThetaDiag' in param_dict.keys():
            JJlin = dA_eta_xi #pxnx2x1
        else:# add X^2/2 suff stat into lin part
            JJlin = dA# pxnx3x1
            
        Ykron=torch.kron(torch.eye(2).unsqueeze(0).unsqueeze(0),Y.expand(p,r,n).permute(0,2,1).unsqueeze(-2))#pxnx2x2r    WARNING MEMORY INTENSIVE
        JJreg=torch.matmul(Ykron.transpose(-2,-1),dA_eta_xi)#pxnx2rx1
        
        XH    =torch.stack((X,H),-1)#pxnx2
        XHkron=torch.kron(torch.eye(2).unsqueeze(0).unsqueeze(0),XH.unsqueeze(-2))#pxnx2x4    WARNING MEMORY INTENSIVE
        kronXH=torch.kron(XH.unsqueeze(-2),torch.eye(2).unsqueeze(0).unsqueeze(0))#pxnx2x4    WARNING MEMORY INTENSIVE
        
        DjWjl=torch.matmul(XHkron.unsqueeze(0).transpose(-2,-1),dA_eta_xi.unsqueeze(1))# pxpxnx4x1
        DlWjl=torch.matmul(kronXH.unsqueeze(0).transpose(-2,-1),dA_eta_xi.unsqueeze(1))# pxpxnx4x1
        JJint=DjWjl+DlWjl.transpose(0,1)# pxpxnx4x1
        
        # concatenate the components
        UU=torch.cat([
                    torch.flatten(JJlin.transpose(0,1),1),#nxp*3
                    torch.flatten(JJreg[pattern_dict['vs'].squeeze()].transpose(0,1),1),            #nxvs*2
                    torch.flatten(JJint[pattern_dict['ci'].triu(1)].transpose(0,1),1)     #nxci*4
                    ],1)# nxdim
        return torch.matmul(UU.t(),UU)*p*p/n
      
    @classmethod
    def dim(cls,variability_matrix,inv_sensitivity_matrix):
        return torch.trace(inv_sensitivity_matrix@variability_matrix)# 
    @classmethod
    def dim_count(cls,param_dict,pattern_dict):
        p,r=param_dict['Gamma'].shape
        selected_params =pattern_dict['vs'].sum()*r*2+pattern_dict['ci'].triu(1).sum()*4# free parameters
        return selected_params+2*p+(p if 'ThetaDiag' in param_dict.keys() else 0)
    @classmethod
    def AIC(cls,ell,dim,p,n):
        return 2*ell*p*n+2*dim
    @classmethod
    def BIC(cls,ell,dim,p,n):
        return 2*ell*p*n+dim*torch.tensor(n).log()
    def compute_AIC_BIC(self,X,H,Y,param_dict,pattern_dict,count_dim=False):
        p,n=X.size()
        ell_pattern=self.ell(X,H,Y,{k:v if k in ('eta','xi','ThetaDiag') else (torch.where(pattern_dict['vs'].reshape(p,1),v,torch.zeros(1)) if k in ['Gamma','Psi'] else torch.where(pattern_dict['ci'],v,torch.zeros(1))) for k,v in param_dict.items()})
            
        if not count_dim:
            variability_matrix=self.estimate_variability_matrix(X,H,Y,param_dict,pattern_dict)
            inv_sensitivity_matrix=self.estimate_inv_sensitivity_matrix(X,H,Y,param_dict,pattern_dict)            
            dim=self.dim(variability_matrix,inv_sensitivity_matrix)
            returns = dict(variability_matrix=variability_matrix,inv_sensitivity_matrix=inv_sensitivity_matrix)
        else:
            dim=self.dim_count(param_dict,pattern_dict)
            returns = {}
        AIC=self.AIC(ell_pattern,dim,p,n)
        BIC=self.BIC(ell_pattern,dim,p,n)
        return AIC,BIC,{'ell':-ell_pattern*p*n,'dim':dim,**returns}
    @classmethod
    def asymptotic_variance(cls,variability_matrix,sensitivity_matrix):
        Hinv=torch.linalg.inv(sensitivity_matrix)
        return Hinv @ variability_matrix @ Hinv
    @classmethod
    def _estimate_dimension_reduction_wch2(cls,reg,aVar,alpha):
        #Bura and Yang [2011]
        p,r=reg.size()
        V_T,D,L=torch.svd(reg,some=False)
        for m in range(min(p,r)+1):
            stat1 = (D[m:].pow(2).sum().mul(n))
            # (1-alpha) quantile
            V0=V_T[:,m:]
            L0=L[m:,:].t()
            kron=torch.kron(L0,V0)
            Q=kron.t().mm(aVar).mm(kron)
            #torch.eig(Q)
            w_ascensing,_=torch.symeig(Q)
            
            s=min(int(torch.matrix_rank(aVar, symmetric=True)),int((p-m)*(r-m)))
            w_s = w_ascensing.flip(0)[:s]

            chi2 = torch.distributions.Chi2(torch.ones(s,10000))
            chi2_s_reps=chi2.sample().double()  # Chi2 distributed with shape df=1
            chi2_w_s_reps=w_s[:,None]*chi2_s_reps
            chi2_w_sums  =chi2_w_s_reps.sum(0)
            # q_alpha=np.quantile(chi2_w_sums.detach().cpu().numpy(),1-alpha)
            q_alpha=torch.quantile(chi2_w_sums,1-alpha)
            if stat1<=q_alpha:
                break
            return m
    @classmethod
    def _estimate_dimension_reduction_ch2(cls,reg,aVar,alpha):
        #Bura and Yang [2011]
        p,r=reg.size()
        V_T,D,L=torch.svd(reg,some=False)
        for m in range(min(p,r)+1):
            V0=V_T[:,m:]
            L0=L[m:,:].t()
            kron=kronecker(L0,V0)
            if kron.numel()==0:
                stat2=0.
            else:
                Q=kron.t().mm(aVar_regression).mm(kron)
                Q_pinv = Q.pinverse()

                D0=torch.zeros(p-m,r-m)
                D0[np.diag_indices(min(p-m,r-m))]=D[m:]
                vecD0=D0.view(-1)
                
                stat2 = vecD0[None,:].mm(Q_pinv).mv(vecD0).mul(n).squeeze()
            # 1- alpha quantile
            s2=min(int(torch.matrix_rank(reg)),int((p-m)*(r-m)))
            q2_alpha=chi2.ppf(1-alpha,s2)
            if stat2<=q2_alpha:
                break
        return m
    def estimate_dimension_reduction(self,param_dict,pattern_dict,asymptotic_variance,alpha=.9,wch2=False):
        p,r = param_dict['Gamma'].size()
        dim_lin=2*p + p if 'ThetaDiag' in param_dict.keys() else 0
        dim_reg=pattern_dict['vs'].sum()*r*2
        aVar_regression=asymptotic_variance[dim_lin:dim_lin+dim_reg,:][:,dim_lin:dim_lin+dim_reg]
        
        if wch2:
            estimate_dim=self._estimate_dimension_reduction_wch2
        else:
            estimate_dim=self._estimate_dimension_reduction_ch2
        
        if self.joint:
            reg=torch.cat((param_dict['Gamma'][pattern_dict['vs']],param_dict['Psi'][pattern_dict['vs']]),0)
            dim=dict(d=estimate_dim(reg,aVar_regression,alpha))
            
        else:
            aVar_Gamma= aVar_regression[:dim_reg//2,:][:,:dim_reg//2]
            aVar_Psi  = aVar_regression[dim_reg//2:dim_reg,:][:,dim_reg//2:dim_reg]
            dim=dict(dG=estimate_dim(param_dict['Gamma'],aVar_Gamma,alpha),
                     dP=estimate_dim(param_dict['Psi'],aVar_Psi,alpha))
            
        return dim
            
        
class HP(HM):
    @classmethod
    def _dAn0(cls,eta,theta=None):
        mu=eta.exp()
        dA=mu# approx when eta is big
        valid_index=eta<2
        beta=(mu[valid_index].exp()-1)/mu[valid_index]-1
        dA[valid_index]+=1/(1+beta)
        return torch.stack((dA,torch.zeros_like(eta),torch.zeros_like(eta)),-1)# pxnx3 eta,xi,thetadiag
    @classmethod
    def _ddAn0(cls,eta,theta=None):
        mu=eta.exp()
        ddA=mu# approx when eta is big
        
        valid_index=eta<2
        beta=(mu[valid_index].exp()-1)/mu[valid_index]-1
        ddA[valid_index]*=(1-1/(1+beta)*(1-beta/(mu[valid_index]*(1+beta))))
        
        return  torch.stack((
                torch.stack((ddA                  ,torch.zeros_like(eta), torch.zeros_like(eta)),-1),
                torch.stack((torch.zeros_like(eta),torch.zeros_like(eta), torch.zeros_like(eta)),-1),
                torch.stack((torch.zeros_like(eta),torch.zeros_like(eta), torch.zeros_like(eta)),-1)
                ),-1)

    @classmethod
    def project_Theta(cls,Theta,ThetaDiag=None):
        return  torch.clamp(Theta,max=0),None

    @classmethod
    def _Ano0(self,eta_cond):
        return eta_cond.exp()+torch.log(1-torch.igammac(torch.tensor(1.),eta_cond.exp()))
        
    @classmethod
    def _conditionals(cls,X,H,Y,param_dict):
        if set(['Gamma','Psi','Phi','PhiT','Theta','Lambda']).issubset(set(param_dict.keys())):
            xi_cond = torch.addmm(param_dict['xi'],param_dict['Psi'],Y) + torch.mm(param_dict['Phi'],X) + torch.mm(param_dict['Lambda'],H)
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y) + torch.mm(param_dict['PhiT'],H) + torch.mm(param_dict['Theta'],X)
            #theta=param_dict['Theta'].diag()
        elif set(['Gamma','Psi']).issubset(set(param_dict.keys())) and not any([k in ('Phi','PhiT','Theta','Lambda') for k in param_dict.keys()]):# indep with regression
            xi_cond = torch.addmm(param_dict['xi'],param_dict['Psi'],Y)
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y)
        elif not any([k in ('Gamma','Psi','Phi','PhiT','Theta','Lambda') for k in param_dict.keys()]):# indept model
            xi_cond =param_dict['xi']
            eta_cond=param_dict['eta']
        else:
            raise AssertionError('invalid set of parameters in param_dict')
        Ano0=cls._Ano0(eta_cond)
        
        #Ahurdle=torch.log(1+(xi_cond+Ano0).exp())
        xiAno0=xi_cond+Ano0
        overflows=xiAno0>(torch.log(torch.tensor(torch.finfo(torch.get_default_dtype()).max))-10)
        if overflows.any():# approx
            Ahurdle=torch.zeros_like(xiAno0)
            Ahurdle[~overflows]=torch.log(1+xiAno0[~overflows].exp())
            Ahurdle[overflows]=xiAno0[overflows]
        else:
            Ahurdle=torch.log(1+xiAno0.exp())
        return dict(xi=xi_cond,eta=eta_cond,A=Ahurdle,Ano0=Ano0)
    @classmethod
    def ell(cls,X,H,Y,param_dict):
        COND = cls._conditionals(X,H,Y,param_dict)
        S=-(H*COND['xi']+COND['eta']*X-torch.lgamma(X+1)-COND['A'])
        return S[~S.isinf()].mean()
    @classmethod
    def _conditional_sample(cls,COND):
        H=torch.distributions.bernoulli.Bernoulli(torch.sigmoid(COND['xi']+COND['Ano0'])).sample()
        nozero=(H==1.)
        X=torch.zeros_like(H)
        X[nozero]=sample_tpoisson(COND['eta'][nozero].exp())
        return X,H
    @classmethod
    def init_dict(cls,X,H,Y):
        # X must be the transformed variable!        
        mean=torch.clamp(torch.mean(X,1,keepdims=True),min=1e-3)
        eta=mean.log()
        # binary
        Ano0=cls._Ano0(eta)
        pinoX=torch.clamp(torch.mean(H,1,keepdims=True),min=.1,max=.9)
        xinoX=torch.log(pinoX/(1-pinoX))
        xinoX[xinoX.isinf()]=torch.abs(Ano0[xinoX.isinf()])*10
        xi=xinoX+Ano0
        return dict(eta=eta,xi=xi)

class HTP(HM):
    def _dAn0(self,eta,theta=None):
        mu=eta.exp()

        P0=torch.exp(-mu)
        P1=P0*mu
        PTstarm1=(-mu+(self.Tstar-1)*eta-torch.lgamma(self.Tstar)).exp()#/factorial(self.Tstar-1)
        P0_over_PTstarm1=(torch.lgamma(self.Tstar)-(self.Tstar-1)*eta).exp()
        
        # beta = (torch.igammac(self.Tstar+1,mu)-P0-P1)/P1# small for mu < 1.01, infty where it is nan
        # beta[mu<1.01]=0
        # beta[beta.isnan()]=float('inf')
        beta = torch.igammac(self.Tstar+1,mu)/P1-1/mu-1# small for mu <= 1
        beta[beta.isnan()]=float('inf')
        
        # alpha= (torch.igammac(self.Tstar-1,mu)-P0)/PTstarm1# small for mu>Tstar and big for mu<<Tstar
        # ALPHA GOES TO ZERO WHEN MU->INFTY
        alpha= torch.igammac(self.Tstar-1,mu)/PTstarm1-P0_over_PTstarm1# small for mu>Tstar and big for mu~Tstar
        alpha[mu>2*self.Tstar]=0
        alpha[alpha.isnan()]=float('inf')
        
        dA=1/(1+beta)+1/(1/mu+1/(self.Tstar*(1+alpha)))
       
        return torch.stack((dA,torch.zeros_like(eta),torch.zeros_like(eta)),-1)# pxnx3 eta,xi,thetadiag

    def _ddAn0(self,eta,theta=None):
        mu=eta.exp()
        
        P0=torch.exp(-mu)
        P1=P0*mu
        PTstarm1=(-mu+(self.Tstar-1)*mu.log()-torch.lgamma(self.Tstar)).exp()#/factorial(self.Tstar-1)
        P0_over_PTstarm1=(torch.lgamma(self.Tstar)-(self.Tstar-1)*eta).exp()
        
        beta = torch.igammac(self.Tstar+1,mu)/P1-1/mu-1# small for mu <= 1
        beta[beta.isnan()]=float('inf')
        
        alpha= torch.igammac(self.Tstar-1,mu)/PTstarm1-P0_over_PTstarm1
        alpha[mu>2*self.Tstar]=0
        alpha[alpha.isnan()]=float('inf')
        dbeta =1-((self.Tstar-1)*mu.log()-torch.lgamma(self.Tstar+1)).exp()+(1-1/mu)*beta
        dbeta[dbeta.isnan()]=float('inf')
        dalpha=(torch.lgamma(self.Tstar)-(self.Tstar-1)*mu.log()).exp()-(self.Tstar-1)/mu+(1-(self.Tstar-1)/mu)*alpha
        dalpha[dalpha.isnan()]=-float('inf')
        
        beta_index=mu<self.Tstar/2
        alpha_index=mu>self.Tstar/2
        ddA=(torch.where(beta_index,-(1+beta).pow(-2)*dbeta,torch.zeros(1))+torch.where(alpha_index,(1/mu+1/(self.Tstar*(1+alpha))).pow(-2)*(mu.pow(-2)+(1+alpha).pow(-2)/self.Tstar*dalpha),torch.ones(1)))*mu

        return  torch.stack((
                torch.stack((ddA                  ,torch.zeros_like(eta), torch.zeros_like(eta)),-1),
                torch.stack((torch.zeros_like(eta),torch.zeros_like(eta), torch.zeros_like(eta)),-1),
                torch.stack((torch.zeros_like(eta),torch.zeros_like(eta), torch.zeros_like(eta)),-1)
                ),-1)
    
    def __init__(self,Tstar=1e4,**kwargs):
        super(HTP,self).__init__(**kwargs)
        self.Tstar=torch.tensor(Tstar)
    def _Ano0(self,eta_cond):
        return eta_cond.exp()+torch.log(
             torch.igammac(self.Tstar+1.,eta_cond.exp())
            -torch.igammac(torch.tensor(1.),eta_cond.exp()))
    def _conditionals(self,X,H,Y,param_dict):
        if set(['Gamma','Psi','Phi','PhiT','Theta','Lambda']).issubset(set(param_dict.keys())):
            xi_cond = torch.addmm(param_dict['xi'],param_dict['Psi'],Y) + torch.mm(param_dict['Phi'],X) + torch.mm(param_dict['Lambda'],H)
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y) + torch.mm(param_dict['PhiT'],H) + torch.mm(param_dict['Theta'],X)
            #theta=param_dict['Theta'].diag()
        elif set(['Gamma','Psi']).issubset(set(param_dict.keys())) and not any([k in ('Phi','PhiT','Theta','Lambda') for k in param_dict.keys()]):# indep with regression
            xi_cond = torch.addmm(param_dict['xi'],param_dict['Psi'],Y)
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y)
        elif not any([k in ('Gamma','Psi','Phi','PhiT','Theta','Lambda') for k in param_dict.keys()]):# indept model
            xi_cond =param_dict['xi']
            eta_cond=param_dict['eta']
        else:
            raise AssertionError('invalid set of parameters in param_dict')

        Ano0=self._Ano0(eta_cond)
        
        xiAno0=xi_cond+Ano0
        # Ahurdle=torch.log(1+xiAno0.exp())
        xiAno0=xi_cond+Ano0
        overflows=xiAno0>(torch.log(torch.tensor(torch.finfo(torch.get_default_dtype()).max))-10)
        if overflows.any():# approx
            Ahurdle=torch.zeros_like(xiAno0)
            Ahurdle[~overflows]=torch.log(1+xiAno0[~overflows].exp())
            Ahurdle[overflows]=xiAno0[overflows]
        else:
            Ahurdle=torch.log(1+xiAno0.exp())
        
        return dict(xi=xi_cond,eta=eta_cond,A=Ahurdle,Ano0=Ano0)
    def ell(self,X,H,Y,param_dict):
        COND = self._conditionals(X,H,Y,param_dict)
        S=-(H*COND['xi']+COND['eta']*X-torch.lgamma(X+1)-COND['A'])
        return S[~S.isinf()].mean()

    def _conditional_sample(self,COND):
        H=torch.distributions.bernoulli.Bernoulli(torch.sigmoid(COND['xi']+COND['Ano0'])).sample()
        nozero=(H==1.)
        X=torch.zeros_like(H)
        X[nozero]=sample_tTpoisson(COND['eta'][nozero].exp(),self.Tstar)
        return X,H
    def transform_data(self,X,H,Y,y=None):
        return X.clamp_max(self.Tstar),H,Y,y
    def init_dict(self,X,H,Y):
        # X must be the transformed variable!
        mean=torch.clamp(torch.mean(X,1,keepdims=True),min=1e-3)
        eta=mean.log()
        # binary
        Ano0=self._Ano0(eta)
        pinoX=torch.clamp(torch.mean(H,1,keepdims=True),min=.1,max=.9)
        xinoX=pinoX/(1-pinoX)
        xinoX[xinoX.isinf()]=torch.abs(Ano0[xinoX.isinf()])*10
        xi=xinoX+Ano0
        return dict(eta=eta,xi=xi)

class HN(HM):
    @classmethod
    def _dAn0(cls,eta,theta):
        return torch.stack((-eta/theta,torch.zeros_like(eta),eta.pow(2)/theta.pow(2)/2-1/theta/2),-1)# pxnx2
    @classmethod
    def _ddAn0(cls,eta,theta):
        return  torch.stack((
                torch.stack((-1/theta.expand_as(eta),torch.zeros_like(eta),eta/theta.pow(2)),-1),
                torch.stack((torch.zeros_like(eta)  ,torch.zeros_like(eta), torch.zeros_like(eta)),-1),
                torch.stack((eta/theta.pow(2)       ,torch.zeros_like(eta),-eta.pow(2)/theta.pow(3)+1/theta.pow(2)/2),-1)
                ),-1)
    
    # pxnx2x2
    def __init__(self,project_theta_onto_negDef=False,**kwargs):
        super(HN,self).__init__(**kwargs)
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
    def _Ano0(cls,eta_cond,theta):
        return -eta_cond.pow(2)/2/theta-torch.log(-theta/2/pi)/2
    @classmethod
    def _conditionals(cls,X,H,Y,param_dict):
        if set(['Gamma','Psi','Phi','PhiT','Theta','Lambda']).issubset(set(param_dict.keys())):
            xi_cond = torch.addmm(param_dict['xi'],param_dict['Psi'],Y) + torch.mm(param_dict['Phi'],X) + torch.mm(param_dict['Lambda'],H)
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y) + torch.mm(param_dict['PhiT'],H) + torch.mm(param_dict['Theta'],X)
        elif set(['Gamma','Psi']).issubset(set(param_dict.keys())) and not any([k in ('Phi','PhiT','Theta','Lambda') for k in param_dict.keys()]):# indep with regression
            xi_cond = torch.addmm(param_dict['xi'],param_dict['Psi'],Y)
            eta_cond= torch.addmm(param_dict['eta'],param_dict['Gamma'],Y)
        elif not any([k in ('Gamma','Psi','Phi','PhiT','Theta','Lambda') for k in param_dict.keys()]):# indept model
            xi_cond =param_dict['xi']
            eta_cond=param_dict['eta']
        else:
            raise AssertionError('invalid set of parameters in param_dict')
        theta=param_dict['ThetaDiag'].reshape(-1,1)

        # Ano0=-eta_cond.pow(2)/2/theta-torch.log(-theta/2/torch.pi)/2# corrected 21/10/25
        # Ano0=-eta_cond.pow(2)/2/theta-torch.log(-theta/2/pi)/2# corrected 21/10/25
        Ano0=cls._Ano0(eta_cond, theta)
        
        Ahurdle=torch.log(1+(xi_cond+Ano0).exp())
        return dict(xi=xi_cond,eta=eta_cond,theta=theta,A=Ahurdle,Ano0=Ano0)
    @classmethod
    def ell(cls,X,H,Y,param_dict):
        COND = cls._conditionals(X,H,Y,param_dict)
        S=-(H*COND['xi']+COND['eta']*X+COND['theta']*X.pow(2)/2-COND['A'])
        return S[~S.isinf()].mean()#/p/n
    @classmethod
    def _conditional_sample(cls,COND):
        H=torch.distributions.bernoulli.Bernoulli(torch.sigmoid(COND['xi']+COND['Ano0'])).sample()
        nozero=(H==1.)
        X=torch.zeros_like(H)
        
        sigma=(-1/COND['theta']).sqrt().expand_as(COND['eta'])
        mu=-COND['eta']/COND['theta']
        X[nozero]=mu[nozero]+sigma[nozero]*torch.randn(int(nozero.sum()))
        return X,H
    @classmethod
    def find_k(H):
        assert H.to(bool).all(1).any(),'H must have a variable that is always 1, remove samples'
        return int(torch.where(H.to(bool).all(1))[0])
    @classmethod
    def transform_data(cls,X,H,Y,y=None):# agree with sample_count
        Z  = torch.log(X/X.sum(0,keepdims=True))
        Z[~Z.bool()]=torch.randn((~Z.bool()).sum())*1e-3
        Z[~H.bool()]=0
        assert (Z.bool()==X.bool()).all(),'zeros change'
        # Hnok=Znok.bool().to(X.dtype)
        return Z,H,Y,y

    @classmethod
    def init_dict(cls,X,H,Y):
        # X must be the transformed variable!
        mean=torch.mean(X,1,keepdims=True)
        var =torch.clamp(torch.var(X,1,keepdims=True),min=1e-3)
        eta=mean/var
        ThetaDiag=-1/var
        # binary
        # Ano0=-param_dict['eta'].pow(2)/2/param_dict['ThetaDiag']-torch.log(-param_dict['ThetaDiag'])/2
        Ano0=cls._Ano0(eta, ThetaDiag)
        pinoX=torch.clamp(torch.mean(H,1,keepdims=True),min=.1,max=.9)
        xinoX=pinoX/(1-pinoX)
        xinoX[xinoX.isinf()]=torch.abs(Ano0[xinoX.isinf()])*10
        xi=xinoX+Ano0
        return dict(eta=eta,xi=xi,ThetaDiag=ThetaDiag)
