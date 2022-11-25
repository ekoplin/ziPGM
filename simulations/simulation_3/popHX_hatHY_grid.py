# -*- coding: utf-8 -*-
'''
This script 
- load a setting, pop_param and pop_pattern from path
- generate data with seed = rep
- train a model
- save performance
'''
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
import argparse
import pathlib
import json
import time
from copy import deepcopy
from itertools import takewhile
import numpy as np
import pandas as pd
import torch

from HM import HP, HTP, HN
from newton_sdmm_pattern import hessian_fisher, pnopt, pnopt_pattern
from newton_indep_model import newton_indep_model, newton_indep_regression_model

from sklearn.metrics import roc_auc_score
from FPR_FNR import FPR_FNR
from kernel_regression import kernelreg_predict
from SPN import normalized_projection_error, relative_projection_error, subespace_angle
from tensor_dict_saving_support import save

from MBGANsampler import sample, phylogenetic 
from sample_truncated_poisson import sample_tpoisson
# %% parse
parser = argparse.ArgumentParser(description='fast variable selection with hatY model')
parser.add_argument('path',     type    =str,                                  help ='path to data and configuration')# reoquired
parser.add_argument('-rep',     type    =int,                                  help ='repetition index, set seed', required=True)# required
parser.add_argument('-model',   choices =['HN','HP','HTP'],                    help ='hat model', required=True)# required
parser.add_argument('-cuda',    action  ='store_true',                         help ='where run pytorch')
parser.add_argument('-save',    type    =bool,                                 help ='save trained models', default=False)
parser.add_argument('-gridsize',type    =int,                                  help ='size of each grid, counting gridsize^2 evaluations',  default=10)
parser.add_argument('-init',    choices=['PF','PFR'],                          help ='initial point', default='PFR')
parser.add_argument('-warm',    action='store_true',                           help ='warm start filtering variables')
parser.add_argument('-tensorboard',type =bool,                                 help ='save partial results in tensorboard?', default=False)
parser_args = parser.parse_args()
args=vars(parser_args)

# required args
path_pop   = pathlib.Path(args.pop('path'))
cuda       = args.pop('cuda')
model_name = args.pop('model')
rep        = args.pop('rep')
save_models= args.pop('save')
gridsize   = args.pop('gridsize')
init       = args.pop('init')
warm       = args.pop('warm')
tensorboard= args.pop('tensorboard')
if cuda:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

# load setting
with open(path_pop/'setting.json', 'r') as f:
    args = json.load(f)
r                   = args['yDim']
N                   = args['nTrain']
Nval                = args['nVal']
Ntest               = args['nTest']
outcome             = args['yType']
signal              = args['signal']
unscaledNoise       = args['unscaledNoise']
# define local path
path = path_pop/model_name
if not path.is_dir():
    path.mkdir()
# define logger to wwite in file
logger_handler=logging.FileHandler(path/f'rep{rep}.log')
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)

''' configuration files are first searched in path if there is not, it search in parent directory up to 2 generations'''
def find_nearer_file(path,name):
    if (path/name).is_file():    
        path_file = path/name
    elif (path.parent/name).is_file():    
        path_file = path.parent/name
    else:
        assert (path.parent.parent/name).is_file(), f'config file {name} was not found'
        path_file = path.parent.parent/name
    return path_file

# load configuration SummaryWriter
with open(find_nearer_file(path,'args_summary_writer.json'), 'r') as f:
    args_summary_writer=json.load(f)
    
# load configuration 
with open(find_nearer_file(path,'args_hessian_fisher_jacobian.json'), 'r') as f:
    args=json.load(f)
damping             = args.pop('damping')
mineig              = args.pop('mineig')
pseudoFisher        = args.pop('pseudoFisher')
batch_size_jacobian = args.pop('batchSizeJacobian')
batch_size_hessian  = args.pop('batchSizeHessian')

with open(find_nearer_file(path,'args_HTP.json'), 'r') as f:
    args=json.load(f)
if args['TstarUsePercentile']:
    Tstar=None
    Tstar_percentile=args['TstarPercentile']
else:
    Tstar = args['Tstar']

# load pnopt setting
with open(find_nearer_file(path,'args_pnopt.json'),'r') as f:
    args_pnopt=json.load(f)
# load pnopti_pattern setting
with open(find_nearer_file(path,'args_pnopt_pattern.json'),'r') as f:
    args_pnopt_pattern=json.load(f)
# load newton_indep_model setting
with open(find_nearer_file(path,'args_newton_indep_model.json'),'r') as f:
    args_newton=json.load(f)

# load signal index andphylo tree
with open(path_pop.parent/'signal.json','r') as f:
    args_signal=json.load(f)
tree   = args_signal['tree']
levels = args_signal['levels']
level  = args_signal['level']
level_keep   = args_signal['level_keep']


ntot  = max(N)+Nval+Ntest
logger.info('start sampling')
try:
    tree_index=pd.MultiIndex.from_tuples(tree,names=levels)
    np.random.seed(rep)
    torch.manual_seed(rep)
    torch.cuda.manual_seed(rep)
    taxa=phylogenetic(sample(ntot,rep,path_pop.parent/'models'/ f"stool_2_ctrl_generator.h5",1e-4))#1/model_pop_TXrefMean))
    taxa_df=pd.DataFrame(taxa,columns=tree_index)
    
    taxa_group=taxa_df.groupby(list(takewhile(lambda l:l!=level,levels))+[level],axis=1).sum().iloc[:,level_keep]
    # taxa_group=taxa_group[taxa_group.columns[(taxa_group != 0).any()]]
    XX    = torch.tensor(taxa_group.to_numpy().T).to(torch.get_default_dtype())
    HH    = XX.bool().to(torch.get_default_dtype())
    p=XX.size(0)
    SPN    = torch.zeros((XX.size(0),1))            
    if signal=='X':
        SPN[args_signal['signalX']]=1.
        pred           = SPN.t()@XX
    elif signal=='H':
        SPN[args_signal['signalH']]=1.
        pred           = SPN.t()@HH
    else:
        raise AssertionError('known signal')
    pop_pattern_vs = SPN.bool().any(1,keepdim = True)
    
    pred_scaled = (pred-pred.mean(1,keepdim = True))/pred.std(1,keepdim = True)
    if outcome =='binary':
        logits      = torch.sigmoid(pred_scaled)
        YY          = (torch.rand_like(logits)<logits).to(torch.get_default_dtype())
    elif outcome=='continous':
        YY=pred_scaled
    else:
        raise AssertionError('not implemented')
except Exception as ex:
    logger.warning(f"sampler broken with exception {str(ex)}")
# transform data 
def transform_data(XX, HH, model):
    if model.__class__.__name__   == 'HN':
        ZZ            = torch.zeros_like(XX)
        ZZ[HH.bool()] = torch.log(XX/XX.sum(0,keepdim=True))[HH.bool()]# use log transform
    elif model.__class__.__name__ == 'HTP':
        ZZ = XX.clamp_max(model.Tstar)
    else:
        ZZ = XX
    return ZZ

for n in N:
    logger.info(f"start n{n}")
    path_n_rep = path/f'n{n}'/f'rep{rep}'
    if not path_n_rep.is_dir():
        path_n_rep.mkdir(parents=True)
    if (path_n_rep/'results.json').is_file():
        continue
    ############ define hat model
    if model_name   == 'HN':
        model = HN(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
    elif model_name == 'HP':
        model = HP(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
    else:
        if Tstar is None:
            Tstar=np.percentile(XX[:,:n][HH[:,:n].bool()].cpu().numpy(), Tstar_percentile)
        model = HTP(Tstar=Tstar,batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
    ############ transform data
    ZZ = transform_data(XX, HH, model)

    # split data
    Ytrain,Yval,Ytest = torch.split(YY,[max(N),Nval,Ntest],dim=1)
    Ztrain,Zval,Ztest = torch.split(ZZ,[max(N),Nval,Ntest],dim=1)
    Xtrain,Xval,Xtest = torch.split(XX,[max(N),Nval,Ntest],dim=1)
    Htrain,Hval,Htest = torch.split(HH,[max(N),Nval,Ntest],dim=1)
    # take the portion of data
    Z,H,Y         =Ztrain[:,:n],Htrain[:,:n],Ytrain[:,:n]
    X             =Xtrain[:,:n]
    # pop reduction in test to measure discrepancy
    if signal=='Z':
        JR_pop_train = SPN.t().mm(Z)
        JR_pop       = SPN.t().mm(Ztest)
    elif signal == 'P':
        JR_pop_train = SPN.t().mm(P)
        JR_pop       = SPN.t().mm(Ptest) 
    elif signal=='X':
        JR_pop_train = SPN.t().mm(X)
        JR_pop       = SPN.t().mm(Xtest) 
    elif signal=='H':
        JR_pop_train = SPN.t().mm(H)
        JR_pop       = SPN.t().mm(Htest) 
    else:
        raise AssertionError('uncompatible signal')

    if outcome=='binary':
        AUC_TEST_POP = roc_auc_score(Ytest.squeeze().cpu().numpy(),JR_pop.squeeze().cpu().numpy())
        PREDERR_TEST_POP = AUC_TEST_POP
    elif outcome == 'continous':
        MSE_TEST_POP = float((Ytest-kernelreg_predict(JR_pop_train,Y,JR_pop)).pow(2).mean())
        PREDERR_TEST_POP = MSE_TEST_POP
    jr_pen       = lambda JR_hat : normalized_projection_error(JR_hat.squeeze(), JR_pop.squeeze(), 1)
    jr_per       = lambda JR_hat : relative_projection_error(JR_hat.squeeze(), JR_pop.squeeze())
    jr_sa        = lambda JR_hat : subespace_angle(JR_hat.t(), JR_pop.t())
    ############ train
    # train indep model
    if tensorboard:
        sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/'indep',**args_summary_writer)
    else:
        sw=None
    indep_dict = model.init_dict(Z,H,Y)
    indep_dict = newton_indep_model(model=model,X=Z,H=H,Y=Y,indep_params=indep_dict,**args_newton,summary_writer=sw)
    if save_models:
        support_path_n_rep = save(path_n_rep)
        support_path_n_rep.save_dict(indep_dict,'indep_hat')
    indep_dict.update(dict(Gamma=torch.zeros(p,Y.size(0)),Psi=torch.zeros(p,Y.size(0)),Theta=torch.zeros(p,p),PhiT=torch.zeros(p,p),Phi=torch.zeros(p,p),Lambda=torch.zeros(p,p)))
    # construct fisher at independent
    PF  =hessian_fisher(model,param_dict=indep_dict, X=Z, H=H, Y=Y,**dict(fisher=True,burn_in=0,damping=damping,mineig=mineig,maxeig=1e99))
    
    if init=='PF':
        # define init
        init_dict=deepcopy(indep_dict)
        for k in ['Gamma','Psi','Theta','Phi','Lambda']:
            init_dict[k]+=torch.randn_like(indep_dict[k])*1e-6
        # scale params 
        if model.__class__.__name__ == 'HN':
            _iscale = (-1/indep_dict['ThetaDiag'].squeeze()).sqrt()
        else:
            _iscale = torch.clamp_min(indep_dict['eta'].squeeze().exp(),1).sqrt()
        init_dict['Theta']*=torch.ger(1/_iscale,1/_iscale)
        init_dict['Phi']  *=torch.ger(torch.ones_like(_iscale),1/_iscale)
        init_dict['Gamma']*=1/_iscale[:,None]      
        # project onto param space
        init_dict['Theta'].fill_diagonal_(0)
        init_dict['Phi'].fill_diagonal_(0)
        init_dict['Lambda'].fill_diagonal_(0)
        init_dict['Theta']=(init_dict['Theta']+init_dict['Theta'].t())/2
        init_dict['Theta'],_=model.project_Theta(init_dict['Theta'],init_dict['ThetaDiag'] if 'ThetaDiag' in init_dict.keys() else None)
        init_dict['Lambda']=(init_dict['Lambda']+init_dict['Lambda'].t())/2
        init_dict['PhiT']=init_dict['Phi'].t()
    elif init=='PFR':
        if tensorboard:
            sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/'indep_regression',**args_summary_writer)
        else:
            sw=None
        indep_with_regression = {k:v.clone() if k in ('eta','xi','ThetaDiag') else torch.randn(*v.shape)*1e-9 for k,v in indep_dict.items() if k in ('eta','xi','ThetaDiag','Gamma','Psi')}
        newton_indep_regression_model(model=model,X=Z,H=H,Y=Y,indep_params=indep_with_regression,**{**args_newton,**dict(rtol=args_newton['rtol']/100,damping=args_newton['damping']/1)},summary_writer=sw)

        init_dict=deepcopy(indep_with_regression)
        for k in ['Theta','Phi','Lambda']:
            init_dict[k]=torch.randn_like(indep_dict[k])*1e-6
        # scale params 
        if model.__class__.__name__ == 'HN':
            _iscale = (-1/indep_dict['ThetaDiag'].squeeze()).sqrt()
        else:
            _iscale = torch.clamp_min(indep_dict['eta'].squeeze().exp(),1).sqrt()
        init_dict['Theta']*=torch.ger(1/_iscale,1/_iscale)
        init_dict['Phi']  *=torch.ger(torch.ones_like(_iscale),1/_iscale)
        # project onto param space
        init_dict['Theta'].fill_diagonal_(0)
        init_dict['Phi'].fill_diagonal_(0)
        init_dict['Lambda'].fill_diagonal_(0)
        init_dict['Theta']=(init_dict['Theta']+init_dict['Theta'].t())/2
        init_dict['Theta'],_=model.project_Theta(init_dict['Theta'],init_dict['ThetaDiag'] if 'ThetaDiag' in init_dict.keys() else None)
        init_dict['Lambda']=(init_dict['Lambda']+init_dict['Lambda'].t())/2
        # define PhiT param
        init_dict['PhiT']=init_dict['Phi'].t()
    else:
        raise AssertionError('known init')
    
    # KKT lambda max
    def kkt_lambda_max(param_dict):
        J                   = model.cat_jacobian(Z, H, Y, param_dict)
        Jbar                = PF.product(PF.apply(lambda x:1/torch.sqrt(x)),J)
        reginter_norm       = (Jbar['inter'].norm(dim = (1,2),keepdim = True).pow(2)+Jbar['reg'].norm(dim = (1),keepdim = True).pow(2).unsqueeze(-1)).sqrt()
        inter_norm          = Jbar['inter'].norm(dim = 2).sqrt()
        lambda_reginter_max = float(reginter_norm.max())
        lambda_inter_max    = float(inter_norm.max())
        return lambda_reginter_max,lambda_inter_max
    # KKT lambda max on init
    lambda_reginter_max,lambda_inter_max = kkt_lambda_max(indep_dict)
    
    # construct hessian
    time_init_PFH=time.process_time()
    PFH =hessian_fisher(model,param_dict=init_dict, X=Z, H=H, Y=Y,**dict(fisher=pseudoFisher,damping=damping,mineig=mineig,burn_in=1000,maxeig=1e99))
    time_elapsed_PFH = time.process_time() - time_init_PFH
    def train(init_dict,lambda_reginter,lambda_inter,refit=True,summary_writer_pnopt=None,summary_writer_refit=None):
        # sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'pnopt_{lambda_reginter:.6f}_{lambda_inter:.6f}')
        time_init_pnopt=time.process_time()
        pnopt_hat,pnopt_hat_pattern = pnopt(   model=model,FisherPF=PF,FisherObj=deepcopy(PFH) if (args_pnopt['update_hessian_each']<args_pnopt['max_iter'] and args_pnopt['update_hessian_eps']<1) else PFH,
                                                        init_dict=init_dict,
                                                        X=Z, H=H, Y=Y, 
                                                        lambda_reginter = lambda_reginter, lambda_inter = lambda_inter,
                                                        **args_pnopt,
                                                        summary_writer=summary_writer_pnopt)
        time_elapsed_pnopt = time.process_time() - time_init_pnopt
        if not pnopt_hat_pattern['vs'].any() and not pnopt_hat_pattern['ci'].any():
            refitted_hat = indep_dict
            time_elapsed_refit=0.
        elif refit:
            # REFIT 
            r_init_dict=pnopt_hat
            time_init_refit=time.process_time()
            PFHR =hessian_fisher(model,param_dict=r_init_dict, X=Z, H=H, Y=Y,**dict(fisher=pseudoFisher,damping=damping,mineig=mineig,burn_in=1000,maxeig=1e99))
            refitted_hat = pnopt_pattern(model=model,FisherObj=PFHR,
                                          init_dict=r_init_dict,pattern_dict=pnopt_hat_pattern,
                                          X=Z, H=H, Y=Y,
                                          **args_pnopt_pattern,
                                          summary_writer=summary_writer_refit)
            time_elapsed_refit = time.process_time() - time_init_refit
        else:
            refitted_hat      = pnopt_hat
            time_elapsed_refit=0.
        return pnopt_hat_pattern, pnopt_hat, refitted_hat, time_elapsed_PFH+time_elapsed_pnopt, time_elapsed_refit    
    
    results                 = {}
    for k in ['n','rep',
              'lambda_reginter_max','lambda_inter_max',
              'lambda_reginter','lambda_inter',
              'i_lambda_reginter','i_lambda_inter',
              'time_pnopt','time_refit',
              'FPR_VS','FNR_VS',
              'PREDERR_TEST','PREDERR_TEST_pop','PREDERR_VAL',
              'AIC','BIC',
              'JR_PEN','JR_PER','JR_SA',
              ]:
        results[k] = []
    train_init = init_dict
    if warm:
        init_inter = init_dict
    for i_lambda_inter, lambda_inter in enumerate(np.linspace(lambda_inter_max,0,num=gridsize)):
        for i_lambda_reginter, lambda_reginter in enumerate(np.linspace(0,lambda_reginter_max,num=gridsize)):
            logger.info(f"start i_lambda_reginter{i_lambda_reginter}_i_lambda_inter{i_lambda_inter}") 
            try:
                if tensorboard:
                    sw =torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'pnopt_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                    swr=torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'pnopt_pattern_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                else:
                    sw=None
                    swr=None
                if warm and i_lambda_reginter==0:
                    train_init = init_inter
                pattern_hat, pnopt_hat, refitted_hat, time_pnopt, time_refit = train(train_init,lambda_reginter,lambda_inter,refit=True,summary_writer_pnopt=sw,summary_writer_refit=swr)
                if warm:
                    train_init = pnopt_hat
                    if i_lambda_reginter==0:
                        init_inter = pnopt_hat
                    if not  pattern_hat['vs'].any():
                        break
                if save_models:
                    support_path_n_rep = save(path_n_rep)
                    support_path_n_rep.save_dict(refitted_hat,f'refited_hat_{i_lambda_reginter}_{i_lambda_inter}')
                    support_path_n_rep.save_dict(pattern_hat,f'pattern_hat_{i_lambda_reginter}_{i_lambda_inter}')
                    support_path_n_rep.save_dict(pnopt_hat,f'pnopt_hat_{i_lambda_reginter}_{i_lambda_inter}')
                   
                # evaluate performance
                FPR_FNR_vs   = FPR_FNR(pattern_hat['vs'].squeeze(),pop_pattern_vs.squeeze())
                # AUC in test
                Rtrain       = refitted_hat['Gamma'].t().mm(Z)+refitted_hat['Psi'].t().mm(H)
                Rtest        = refitted_hat['Gamma'].t().mm(Ztest)+refitted_hat['Psi'].t().mm(Htest)
                Rval         = refitted_hat['Gamma'].t().mm(Zval)+refitted_hat['Psi'].t().mm(Hval)
                if outcome=='binary':
                    AUC_TEST = roc_auc_score(Ytest.t().cpu().numpy(),Rtest.t().cpu().numpy())
                    AUC_VAL  = roc_auc_score(Yval.t().cpu().numpy(),Rval.t().cpu().numpy())
                    PREDERR_TEST = AUC_TEST
                    PREDERR_VAL  = AUC_VAL
                elif outcome=='continous':
                    YRtest = kernelreg_predict(Rtrain,Y,Rtest)
                    YRval  = kernelreg_predict(Rtrain,Y,Rval)
                    MSE_TEST = float((Ytest[~YRtest.isnan()]-YRtest[~YRtest.isnan()]).pow(2).mean())
                    MSE_VAL  = float((Ytest[~YRval.isnan()]-YRval[~YRval.isnan()]).pow(2).mean())
                    PREDERR_TEST = MSE_TEST
                    PREDERR_VAL  = MSE_VAL                  
                # evaluate criterions
                AIC,BIC,_             = model.compute_AIC_BIC(Z,H,Y,refitted_hat,pattern_hat,count_dim = True)
                # SPN
                JR_hat                = refitted_hat['Gamma'].t().mm(Ztest)+refitted_hat['Psi'].t().mm(Htest)
                JR_PEN                = jr_pen(JR_hat)
                JR_PER                = jr_per(JR_hat)
                JR_SA                 = jr_sa(JR_hat)
                # save some results
                if tensorboard:
                    swp =torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'results_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                    swp.add_scalars('selection/vs',dict(zip(['FPR','FNR'],FPR_FNR_vs)))
                    swp.add_scalar('criterion/PREDERR_val', PREDERR_VAL)
                    swp.add_scalar('criterion/AIC', AIC)
                    swp.add_scalar('criterion/BIC', BIC)
                    swp.add_scalar('performance/PREDERR_TEST', PREDERR_TEST)
                    swp.add_scalar('performance/JR_PEN', JR_PEN)
            except Exception as ex:
                logger.warning(f"train broken {str(ex)}")
                time_pnopt    = np.nan
                time_refit    = np.nan
                FPR_FNR_vs    = np.nan,np.nan
                # AUC in test
                PREDERR_TEST  = np.nan
                # evaluate cr
                PREDERR_VAL   = np.nan
                AIC,BIC       = np.nan, np.nan
                # SPN
                JR_PEN        = np.nan
                JR_PER        = np.nan
                JR_SA         = np.nan
            # save results
            results['n'].append(n)
            results['rep'].append(rep)
            results['lambda_reginter_max'].append(lambda_reginter_max)
            results['lambda_inter_max'].append(lambda_inter_max)
            results['lambda_reginter'].append(lambda_reginter)
            results['lambda_inter'].append(lambda_inter)
            results['i_lambda_reginter'].append(i_lambda_reginter)
            results['i_lambda_inter'].append(i_lambda_inter)
            results['time_pnopt'].append(time_pnopt)
            results['time_refit'].append(time_refit)
            results['FPR_VS'].append(FPR_FNR_vs[0])
            results['FNR_VS'].append(FPR_FNR_vs[1])
            results['PREDERR_TEST'].append(PREDERR_TEST)
            results['PREDERR_TEST_pop'].append(PREDERR_TEST_POP)
            results['PREDERR_VAL'].append(PREDERR_VAL)
            results['AIC'].append(float(AIC))
            results['BIC'].append(float(BIC))
            results['JR_PEN'].append(JR_PEN)
            results['JR_PER'].append(JR_PER)
            results['JR_SA'].append(JR_SA)
    with open(path_n_rep/('results.json'),'w') as f:
        json.dump(results,f)
    logger.info('end')
