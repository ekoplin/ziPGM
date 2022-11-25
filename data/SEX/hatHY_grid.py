# -*- coding: utf-8 -*-
'''
This script 
- load the filtered data, 
- select fold
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

from HM import HP, HTP, HN
from newton_sdmm_pattern import hessian_fisher, pnopt, pnopt_pattern
from newton_indep_model import newton_indep_model, newton_indep_regression_model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score#,log_loss

from tensor_dict_saving_support import save

# %% parse
parser = argparse.ArgumentParser(description='fast variable selection with hatY model')
parser.add_argument('path',     type    =str,                                  help ='path to data and configuration')# reoquired
parser.add_argument('-fold',    type    =int, choices=range(0,5),              help ='5-fold index', required=True)# required
parser.add_argument('-ifold',   type    =int, choices=range(-1,5),             help ='5-(inner)fold index, -1 means train with all the data', default=-1)# required
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

# outcome    = args.pop('outcome')
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
df=pd.read_csv(path_pop/'HGP_SEX.csv',header=[0,1],index_col=0)
XX=torch.tensor(df.Taxa.to_numpy().T)
HH=XX.bool().to(torch.get_default_dtype())
p = XX.size(0)
YY=torch.tensor(pd.Categorical(df.Outcome.SEX,categories=['male','female']).codes.astype('double').reshape(1,-1))
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
    indx_k=None
    if model.__class__.__name__   == 'HN':
        indx_k   = int(XX.bool().sum(1).argmax())
        indx_nok = [i!=indx_k for i in range(p)] 
        HH=HH[indx_nok,:]
        HH[:,~XX[indx_k].bool()]=0
        ZZ,_          = torch.split(torch.zeros_like(XX),[p-1,1])
        ZZ[HH.bool()] = torch.log(XX[indx_nok,:]/XX[indx_k,:])[HH.bool()]# use log transform
    elif model.__class__.__name__ == 'HTP':
        ZZ = XX.clamp_max(model.Tstar)
    else:
        ZZ = XX
    return ZZ,HH,indx_k


logger.info("start training")
path_fold_ifold = path/f'fold{fold}'/f'ifold{ifold}'
if not path_fold_ifold.is_dir():
    path_fold_ifold.mkdir(parents=True)
if (path_fold_ifold/'results.json').is_file():
    exit()
############ define hat model
if model_name   == 'HN':
    model = HN(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
elif model_name == 'HP':
    model = HP(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
else:
    if Tstar is None:
        Tstar=np.percentile(XX[:,indx_train][HH[:,indx_train].bool()].cpu().numpy(), Tstar_percentile)
    model = HTP(Tstar=Tstar,batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
############ transform data
ZZ,HH,indx_k = transform_data(XX, HH, model)

# split data
Y,Ytest = YY[:,indx_train],YY[:,indx_test]
Z,Ztest = ZZ[:,indx_train],ZZ[:,indx_test]
X,Xtest = XX[:,indx_train],XX[:,indx_test]
H,Htest = HH[:,indx_train],HH[:,indx_test]

p=ZZ.size(0)
############ train
if ifold==-1:# train indep and compute lambda max
    # train indep model
    if tensorboard:
        sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/'indep',**args_summary_writer)
    else:
        sw=None
    indep_dict = model.init_dict(Z,H,Y)
    indep_dict = newton_indep_model(model=model,X=Z,H=H,Y=Y,indep_params=indep_dict,**args_newton,summary_writer=sw)
    if save_models:
        support_path_fold_ifold = save(path_fold_ifold)
        support_path_fold_ifold.save_dict(indep_dict,'indep_hat')
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
            sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/'indep_regression',**args_summary_writer)
        else:
            sw=None
        indep_with_regression = {k:v.clone() if k in ('eta','xi','ThetaDiag') else torch.randn(*v.shape)*1e-1 for k,v in indep_dict.items() if k in ('eta','xi','ThetaDiag','Gamma','Psi')}
        newton_indep_regression_model(model=model,X=Z,H=H,Y=Y,indep_params=indep_with_regression,**{**args_newton,**dict(rtol=args_newton['rtol']/100,damping=args_newton['damping']*1000,step=args_newton['step']/1000)},summary_writer=sw)
    
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
        init_dict['PhiT']=init_dict['Phi'].t()
    else:
        raise AssertionError('known init')
    
    # KKT lambda max
    def kkt_lambda_max(param_dict):
        J                   = model.cat_jacobian(Z, H, Y, param_dict)
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
    # save PF????
    with open(path_fold_ifold/'lambda_max.json','w') as f:
        json.dump(dict(lambda_reginter_max=lambda_reginter_max,lambda_inter_max=lambda_inter_max), f)
else:
    # load
    support_path_fold_ifold = save(path_fold_ifold.parent/'ifold-1')
    indep_dict=support_path_fold_ifold.load_dict('indep_dict')
    init_dict=support_path_fold_ifold.load_dict('init_dict')
    
    PF  =hessian_fisher(model,param_dict=indep_dict, X=Z, H=H, Y=Y,**dict(fisher=True,burn_in=0,damping=damping,mineig=mineig,maxeig=1e99))
    with open(path_fold_ifold.parent/'ifold-1'/'lambda_max.json','r') as f:
        lambda_max_dict=json.load(f)
        lambda_reginter_max=lambda_max_dict.pop('lambda_reginter_max')
        lambda_inter_max=lambda_max_dict.pop('lambda_inter_max')
    
# construct hessian
time_init_PFH=time.process_time()
PFH =hessian_fisher(model,param_dict=init_dict, X=Z, H=H, Y=Y,**dict(fisher=pseudoFisher,damping=damping,mineig=mineig,burn_in=1000,maxeig=1e99))
time_elapsed_PFH = time.process_time() - time_init_PFH
def train(init_dict,lambda_reginter,lambda_inter,refit=True,summary_writer_pnopt=None,summary_writer_refit=None):
    # sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/f'pnopt_{lambda_reginter:.6f}_{lambda_inter:.6f}')
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
for k in ['fold','ifold',
          'lambda_reginter_max','lambda_inter_max',
          'lambda_reginter','lambda_inter',
          'i_lambda_reginter','i_lambda_inter',
          'time_pnopt','time_refit',
          'PREDERR_TEST','PREDERR_TEST_S',
          'SELECTED_VARS'
          ]:
    results[k] = []
train_init = init_dict
if warm:
    init_inter = init_dict
for i_lambda_inter, lambda_inter in enumerate(np.geomspace(lambda_inter_max,lambda_inter_max/gridsize,num=gridsize)):
    for i_lambda_reginter, lambda_reginter in enumerate(np.linspace(lambda_reginter_max/gridsize,lambda_reginter_max,num=gridsize)):

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
            #joint
            Rtrain       = refitted_hat['Gamma'].t().mm(Z)+refitted_hat['Psi'].t().mm(H)
            Rtest        = refitted_hat['Gamma'].t().mm(Ztest)+refitted_hat['Psi'].t().mm(Htest)
            #separated
            Rtrain_S       = torch.cat([refitted_hat['Gamma'].t().mm(Z),refitted_hat['Psi'].t().mm(H)],0)
            Rtest_S        = torch.cat([refitted_hat['Gamma'].t().mm(Ztest),refitted_hat['Psi'].t().mm(Htest)],0)
            if model_name=='HN':# add reference var into the reduction
                SELECTED_VARS.insert(indx_k,True)
            
                Rtrain=torch.cat([Rtrain,X[None,indx_k,:]],0)
                Rtest=torch.cat([Rtest,Xtest[None,indx_k,:]],0)
                Rtrain_S=torch.cat([Rtrain_S,X[None,indx_k,:]],0)
                Rtest_S=torch.cat([Rtest_S,Xtest[None,indx_k,:]],0)
                
                clf = LogisticRegression(random_state=0).fit(Rtrain.cpu().numpy().T,Y.squeeze().cpu().numpy()>0)
                AUC_TEST = roc_auc_score(Ytest.t().cpu().numpy()>0,clf.predict_proba(Rtest.t().cpu().numpy())[:,1])
                clf = LogisticRegression(random_state=0).fit(Rtrain_S.cpu().numpy().T,Y.squeeze().cpu().numpy()>0)
                AUC_TEST_S = roc_auc_score(Ytest.t().cpu().numpy()>0,clf.predict_proba(Rtest_S.t().cpu().numpy())[:,1])
                
            else:
                AUC_TEST = roc_auc_score(Ytest.t().cpu().numpy()>0,Rtest.t().cpu().numpy())
                clf = LogisticRegression(random_state=0).fit(Rtrain_S.cpu().numpy().T,Y.squeeze().cpu().numpy()>0)
                AUC_TEST_S = roc_auc_score(Ytest.t().cpu().numpy()>0,clf.predict_proba(Rtest_S.t().cpu().numpy())[:,1])
            PREDERR_TEST   = AUC_TEST
            PREDERR_TEST_S = AUC_TEST_S
            # save some results
            if tensorboard:
                swp =torch.utils.tensorboard.SummaryWriter(log_dir=path_fold_ifold/f'results_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                swp.add_scalar('performance/PREDERR_TEST', PREDERR_TEST)
                swp.add_scalar('performance/PREDERR_TEST_S', PREDERR_TEST_S)
                swp.add_scalar('performance/sum(SELECTED_VARS)', pattern_hat['vs'].sum())
            if ifold==-1:# save reduction
                support_path_fold_ifold = save(path_fold_ifold)
                support_path_fold_ifold.save_dict({'Rtrain':Rtrain,'Rtest':Rtest,'Rtrain_S':Rtrain_S,'Rtest_S':Rtest_S,'Ytrain':Y,'Ytest':Ytest},f'reductions_{i_lambda_reginter}_{i_lambda_inter}')
        except Exception as ex:
            logger.warning(f"train broken {str(ex)}")
            time_pnopt    = np.nan
            time_refit    = np.nan
            # AUC in test
            PREDERR_TEST  = np.nan 
            PREDERR_TEST_S= np.nan 
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
        results['PREDERR_TEST_S'].append(PREDERR_TEST_S)
        results['SELECTED_VARS'].append(SELECTED_VARS)
        
with open(path_fold_ifold/('results.json'),'w') as f:
    json.dump(results,f)
logger.info('end')
