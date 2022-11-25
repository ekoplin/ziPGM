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

import numpy as np
import pandas as pd
import torch

from PGM import PPGM, TPPGM, NPGM, BPGM
from PGM_newton_sdmm_pattern import PGM_hessian_fisher, PGM_pnopt, PGM_pnopt_pattern
from PGM_newton_indep_model import PGM_newton_indep_model,PGM_newton_indep_regression_model

from sklearn.metrics import roc_auc_score
from kernel_regression import kernelreg_predict

from tensor_dict_saving_support import save
# %% parse
parser = argparse.ArgumentParser(description='fast variable selection with hatY model')
parser.add_argument('path',     type    =str,                                  help ='path to data and configuration')# reoquired
parser.add_argument('-fold',    type    =int, choices=range(0,5),              help ='5-fold index', required=True)# required
parser.add_argument('-ifold',   type    =int, choices=range(-1,5),             help ='5-(inner)fold index, -1 means train with all the data', default=-1)# required
parser.add_argument('-outcome', choices =['binary','continous','log','continous_log_squared','continous_BMI_AGE','continous_logBMI_AGE'], help ='outcome BMI binary, continous or log(continous)', required=True)# required
parser.add_argument('-model',   choices =['NPGM','PPGM','TPPGM','BPGM'],       help ='hat model', required=True)# required
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

outcome    = args.pop('outcome')
fold       = args.pop('fold')
ifold      = args.pop('ifold')
print(fold,ifold)
save_models= args.pop('save')
gridsize   = args.pop('gridsize')
init       = args.pop('init')
warm       = args.pop('warm')
tensorboard= args.pop('tensorboard')

if cuda:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')


# define local path
path = path_pop/model_name
if not path.is_dir():
    path.mkdir()
# define logger to wwite in file
logger_handler=logging.FileHandler(path/f'fold{fold}_ifold{ifold}.log')
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)

''' configuration files are first searched in path if there is not, it search in parent directory up to 2 generations'''
def find_nearer_file(path,name):
    if (path/name).is_file():    
        path_file = path/name
    elif (path.parent/name).is_file():    
        path_file = path.parent/name
    elif (path.parent.parent/name).is_file():    
        path_file = path.parent.parent/name
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

# load data
df=pd.read_csv(path_pop/'HGP_BMI.csv',header=[0,1],index_col=0)
XX=torch.tensor(df.Taxa.to_numpy().T)
HH=XX.bool().to(torch.get_default_dtype())
p = XX.size(0)
if outcome=='binary':
    YY=torch.tensor(df.Outcome.BMIge30.to_numpy().T).reshape(1,-1)
elif outcome=='continous':
    YY=torch.tensor(df.Outcome.BMI.to_numpy().T).reshape(1,-1)
elif outcome=='log':
    YY=torch.tensor(df.Outcome.logBMI.to_numpy().T).reshape(1,-1)
elif outcome=='continous_log_squared':
    YY=torch.tensor([df.Outcome.BMI.to_numpy(),df.Outcome.logBMI.to_numpy(),np.power(df.Outcome.BMI.to_numpy(),2)]).reshape(3,-1)
elif outcome=='continous_BMI_AGE':
    # YY=torch.tensor(df[[('Outcome','logBMI'),('Meta','AGE')]].to_numpy().astype('double').T)
    YY=torch.tensor(df[[('Meta','BMI'),('Meta','AGE')]].to_numpy().astype('double').T)
elif outcome=='continous_logBMI_AGE':
    YY=torch.tensor(df[[('Outcome','logBMI'),('Meta','AGE')]].to_numpy().astype('double').T)
else:
    raise AssertionError(f'unknown outcome {outcome}')

# standarize Y
YY-=YY.mean(1,keepdim=True)
YY/=YY.std(1,keepdim=True)
    
if ifold==-1:
    indx_train = df.Folds.Outer!=fold
    indx_test  = df.Folds.Outer==fold
else:
    indx_train = ((df.Folds.Outer!=fold) & (df.Folds[str(fold)]!=ifold)).to_numpy()
    indx_test  = ((df.Folds.Outer!=fold) & (df.Folds[str(fold)]==ifold)).to_numpy()

# transform data 
def transform_data(XX, HH, model):
    if model.__class__.__name__ == 'BPGM':
        ZZ = HH
    elif model.__class__.__name__   == 'NPGM':
        ZZ            = torch.randn_like(XX)*1e-3#torch.zeros_like(XX)
        ZZ[HH.bool()] = torch.log(XX/XX.sum(0,keepdim=True))[HH.bool()]# use log transform
    elif model.__class__.__name__ == 'TPPGM':
        ZZ = XX.clamp_max(model.Tstar)
    else:
        ZZ = XX
    return ZZ


logger.info("start training")
path_fold_ifold = path/f'fold{fold}'/f'ifold{ifold}'
if not path_fold_ifold.is_dir():
    path_fold_ifold.mkdir(parents=True)
if (path_fold_ifold/'results.json').is_file():
    exit()
############ define hat model
if model_name   == 'BPGM':
    model = BPGM(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
elif model_name   == 'NPGM':
    model = NPGM(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
elif model_name == 'PPGM':
    model = PPGM(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
else:
    if Tstar is None:
        # find percentile using train set
        Tstar=np.percentile(XX[:,indx_train][HH[:,indx_train].bool()].cpu().numpy(), Tstar_percentile)
    model = TPPGM(Tstar=Tstar,batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)

############ transform data
ZZ = transform_data(XX, HH, model)

# split data
Y,Ytest = YY[:,indx_train],YY[:,indx_test]
Z,Ztest = ZZ[:,indx_train],ZZ[:,indx_test]
X,Xtest = XX[:,indx_train],XX[:,indx_test]
H,Htest = HH[:,indx_train],HH[:,indx_test]


############ train
if ifold==-1:# train indep and compute lambda max
    # train indep model
    if tensorboard:
        sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/'indep',**args_summary_writer)
    else:
        sw=None
        
    indep_dict = model.init_dict(Z,Y)
    indep_dict = PGM_newton_indep_model(model=model,X=Z,Y=Y,indep_params=indep_dict,**args_newton,summary_writer=sw)
    if save_models:
        support_path_n_rep = save(path_fold_ifold)
        support_path_n_rep.save_dict(indep_dict,'indep_hat')
    indep_dict.update(dict(Gamma=torch.zeros(p,Y.size(0)),Theta=torch.zeros(p,p)))
    # construct fisher at independent
    PF  =PGM_hessian_fisher(model,param_dict=indep_dict, X=Z, Y=Y,**dict(fisher=True,burn_in=0,damping=damping,mineig=mineig,maxeig=1e99))
    
    if init=='PF':
        # define init
        init_dict=deepcopy(indep_dict)
        for k in ['Gamma','Theta']:
            init_dict[k]+=torch.randn_like(indep_dict[k])*1e-6
        # scale params 
        if model.__class__.__name__ == 'NPGM':
            _iscale = (-1/indep_dict['ThetaDiag'].squeeze()).sqrt()
        elif model.__class__.__name__ in ['PPGM','TPPGM']:
            _iscale = torch.clamp_min(indep_dict['eta'].squeeze().exp(),1).sqrt()
        else:
            _iscale = torch.ones_like(indep_dict['eta'].squeeze())
        init_dict['Theta']*=torch.ger(1/_iscale,1/_iscale)
        init_dict['Gamma']*=1/_iscale[:,None]      
        # project onto param space
        init_dict['Theta'].fill_diagonal_(0)
        init_dict['Theta']=(init_dict['Theta']+init_dict['Theta'].t())/2
        init_dict['Theta'],_=model.project_Theta(init_dict['Theta'],init_dict['ThetaDiag'] if 'ThetaDiag' in init_dict.keys() else None)
    elif init=='PFR':
        if tensorboard:
            sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/'indep_regression',**args_summary_writer)
        else:
            sw=None
        indep_with_regression = {k:v.clone() if k in ('eta','ThetaDiag') else torch.randn(*v.shape)*1e-9 for k,v in indep_dict.items() if k in ('eta','ThetaDiag','Gamma')}
        PGM_newton_indep_regression_model(model=model,X=Z,Y=Y,indep_params=indep_with_regression,**{**args_newton,**dict(rtol=args_newton['rtol']/100,damping=args_newton['damping']*1000,step=args_newton['step']/1000)},summary_writer=sw)
    
        init_dict=deepcopy(indep_with_regression)
        for k in ['Theta']:
            init_dict[k]=torch.randn_like(indep_dict[k])*1e-6
        # scale params 
        if model.__class__.__name__ == 'NPGM':
            _iscale = (-1/indep_dict['ThetaDiag'].squeeze()).sqrt()
        elif model.__class__.__name__ in ['PPGM','TPPGM']:
            _iscale = torch.clamp_min(indep_dict['eta'].squeeze().exp(),1).sqrt()
        else:
            _iscale = torch.ones_like(indep_dict['eta'].squeeze())
        init_dict['Theta']*=torch.ger(1/_iscale,1/_iscale)
        # project onto param space
        init_dict['Theta'].fill_diagonal_(0)
        init_dict['Theta']=(init_dict['Theta']+init_dict['Theta'].t())/2
        init_dict['Theta'],_=model.project_Theta(init_dict['Theta'],init_dict['ThetaDiag'] if 'ThetaDiag' in init_dict.keys() else None)
    else:
        raise AssertionError('known init')
    
    # KKT lambda max
    def kkt_lambda_max(param_dict):
        J                   = model.cat_jacobian(Z, Y, param_dict)
        Jbar                = PF.product(PF.apply(lambda x:1/torch.sqrt(x)),J)
        # reginter_norm       = (Jbar['inter'].norm(dim = (1,2),keepdim = True).pow(2)+Jbar['inter'].norm(dim = (0,2),keepdim = True).pow(2)+Jbar['reg'].norm(dim = (1),keepdim = True).pow(2).unsqueeze(-1)).sqrt()
        # inter_norm          = (Jbar['inter'].norm(dim = 2)+Jbar['inter'].norm(dim = 2).transpose(0,1)).sqrt()
        reginter_norm       = (Jbar['inter'].norm(dim = (1,2),keepdim = True).pow(2)+Jbar['reg'].norm(dim = (1),keepdim = True).pow(2).unsqueeze(-1)).sqrt()
        inter_norm          = Jbar['inter'].norm(dim = 2).sqrt()
        lambda_reginter_max = float(reginter_norm.max())
        lambda_inter_max    = float(inter_norm.max())
        return lambda_reginter_max,lambda_inter_max
    # KKT lambda max on init
    lambda_reginter_max,lambda_inter_max = kkt_lambda_max(indep_dict)
     
    # save init
    support_path_fold_ifold = save(path_fold_ifold)
    support_path_fold_ifold.save_dict(indep_dict,'indep_dict')
    support_path_fold_ifold.save_dict(init_dict,'init_dict')
    # save PF
    with open(path_fold_ifold/'lambda_max.json','w') as f:
        json.dump(dict(lambda_reginter_max=lambda_reginter_max,lambda_inter_max=lambda_inter_max), f)
else:
    # load
    support_path_fold_ifold = save(path_fold_ifold.parent/'ifold-1')
    indep_dict=support_path_fold_ifold.load_dict('indep_dict')
    init_dict=support_path_fold_ifold.load_dict('init_dict')
    # load PF
    PF  =PGM_hessian_fisher(model,param_dict=indep_dict, X=Z,Y=Y,**dict(fisher=True,burn_in=0,damping=damping,mineig=mineig,maxeig=1e99))
    with open(path_fold_ifold.parent/'ifold-1'/'lambda_max.json','r') as f:
        lambda_max_dict=json.load(f)
        lambda_reginter_max=lambda_max_dict.pop('lambda_reginter_max')
        lambda_inter_max=lambda_max_dict.pop('lambda_inter_max')
    
# construct hessian
time_init_PFH=time.process_time()
PFH =PGM_hessian_fisher(model,param_dict=init_dict, X=Z,Y=Y,**dict(fisher=pseudoFisher,damping=damping,mineig=mineig,burn_in=1000,maxeig=1e99))
time_elapsed_PFH = time.process_time() - time_init_PFH
def train(init_dict,lambda_reginter,lambda_inter,refit=True,summary_writer_pnopt=None,summary_writer_refit=None):
    time_init_pnopt=time.process_time()
    pnopt_hat,pnopt_hat_pattern = PGM_pnopt(   model=model,FisherPF=PF,FisherObj=deepcopy(PFH) if (args_pnopt['update_hessian_each']<args_pnopt['max_iter'] and args_pnopt['update_hessian_eps']<1) else PFH,
                                                    init_dict=init_dict,
                                                    X=Z, Y=Y, 
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
        PFHR =PGM_hessian_fisher(model,param_dict=r_init_dict, X=Z, Y=Y,**dict(fisher=pseudoFisher,damping=damping,mineig=mineig,burn_in=1000,maxeig=1e99))
        # swr=torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/f'pnopt_pattern_{lambda_reginter:.6f}_{lambda_inter:.6f}')
        refitted_hat = PGM_pnopt_pattern(model=model,FisherObj=PFHR,
                                      init_dict=r_init_dict,pattern_dict=pnopt_hat_pattern,
                                      X=Z,Y=Y,
                                      **args_pnopt_pattern,
                                      summary_writer=summary_writer_refit)
        time_elapsed_refit = time.process_time() - time_init_refit
    else:
        refitted_hat      = pnopt_hat
        time_elapsed_refit=0.
    return pnopt_hat_pattern, pnopt_hat, refitted_hat, time_elapsed_PFH+time_elapsed_pnopt, time_elapsed_refit    

results                 = {}
for k in ['fold','ifold',
          'lambda_reginter_max','lambda_inter_max',
          'lambda_reginter','lambda_inter',
          'i_lambda_reginter','i_lambda_inter',
          'time_pnopt','time_refit',
          'PREDERR_TEST',
          'SELECTED_VARS'
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
                sw =torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/f'pnopt_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                swr=torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/f'pnopt_pattern_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
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
                support_path_fold_ifold = save(path_fold_ifold)
                support_path_fold_ifold.save_dict(refitted_hat,f'refited_hat_{i_lambda_reginter}_{i_lambda_inter}')
                support_path_fold_ifold.save_dict(pattern_hat,f'pattern_hat_{i_lambda_reginter}_{i_lambda_inter}')
                support_path_fold_ifold.save_dict(pnopt_hat,f'pnopt_hat_{i_lambda_reginter}_{i_lambda_inter}')
               
            # evaluate performance
            SELECTED_VARS= pattern_hat['vs'].squeeze().cpu().numpy().tolist()
            # AUC in test
            if refitted_hat['Gamma'].size(1)>1:
                U,S,V=torch.svd(refitted_hat['Gamma'])
                U*=S
                Rtrain       = U.t().mm(Z)
                Rtest        = U.t().mm(Ztest)
            else:
                Rtrain       = refitted_hat['Gamma'].t().mm(Z)
                Rtest        = refitted_hat['Gamma'].t().mm(Ztest)
            if outcome=='binary':
                AUC_TEST = roc_auc_score(Ytest.t().cpu().numpy()>0,Rtest.t().cpu().numpy())
                PREDERR_TEST = AUC_TEST
            elif 'continous' in outcome:
                YRtest = kernelreg_predict(Rtrain,Y,Rtest)
                MSE_TEST = float((Ytest[~YRtest.isnan()]-YRtest[~YRtest.isnan()]).pow(2).mean())
                PREDERR_TEST = MSE_TEST
            # save some results
            if tensorboard:
                swp =torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/f'results_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                swp.add_scalar('performance/PREDERR_TEST', PREDERR_TEST)
                swp.add_scalar('performance/sum(SELECTED_VARS)', pattern_hat['vs'].sum())
            if ifold==-1:# save reduction
                support_path_fold_ifold = save(path_fold_ifold)
                support_path_fold_ifold.save_dict({'Rtrain':Rtrain,'Rtest':Rtest,'Ytrain':Y,'Ytest':Ytest},f'reductions_{i_lambda_reginter}_{i_lambda_inter}')
        except Exception as ex:
            logger.warning(f"train broken {str(ex)}")
            time_pnopt    = np.nan
            time_refit    = np.nan
            # AUC in test
            PREDERR_TEST  = np.nan     
            SELECTED_VARS = [np.nan]*p
        # save results
        results['fold'].append(fold)
        results['ifold'].append(ifold)
        results['lambda_reginter_max'].append(lambda_reginter_max)
        results['lambda_inter_max'].append(lambda_inter_max)
        results['lambda_reginter'].append(lambda_reginter)
        results['lambda_inter'].append(lambda_inter)
        results['i_lambda_reginter'].append(i_lambda_reginter)
        results['i_lambda_inter'].append(i_lambda_inter)
        results['time_pnopt'].append(time_pnopt)
        results['time_refit'].append(time_refit)
        results['PREDERR_TEST'].append(PREDERR_TEST)
        results['SELECTED_VARS'].append(SELECTED_VARS)
        
with open(path_fold_ifold/('results.json'),'w') as f:
    json.dump(results,f)
logger.info('end')
