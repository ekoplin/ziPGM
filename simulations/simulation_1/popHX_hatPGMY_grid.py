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
import torch

from HM import HP, HTP, HN
from PGM import PPGM, TPPGM, NPGM, BPGM
from PGM_newton_sdmm_pattern import PGM_hessian_fisher, PGM_pnopt, PGM_pnopt_pattern
from PGM_newton_indep_model import PGM_newton_indep_model

from sklearn.metrics import roc_auc_score
from FPR_FNR import FPR_FNR
from SPN import normalized_projection_error, relative_projection_error, subespace_angle
from tensor_dict_saving_support import save

# %% parse
parser = argparse.ArgumentParser(description='fast variable selection with hatY model')
parser.add_argument('path',     type    =str,                       help ='path to data and configuration')# reoquired
parser.add_argument('-rep',     type    =int,                       help ='repetition index, set seed', required=True)# required
parser.add_argument('-model',   choices =['BPGM','NPGM','PPGM','TPPGM'],   help ='hat model', required=True)# required
parser.add_argument('-cuda',    action  ='store_true',              help ='where run pytorch')
parser.add_argument('-save',    type    =bool, default=False,       help ='save trained models')
parser.add_argument('-gridsize',type    =int,  default=10,          help ='repetition index, set seed')
parser.add_argument('-tensorboard',type =bool, default=False,       help ='save partial results in tensorboard?')
parser_args = parser.parse_args()
args=vars(parser_args)

# required args
path_pop   = pathlib.Path(args.pop('path'))
cuda       = args.pop('cuda')
model_name = args.pop('model')
rep        = args.pop('rep')
save_models= args.pop('save')
gridsize   = args.pop('gridsize')
tensorboard= args.pop('tensorboard')
if cuda:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

# load setting
with open(path_pop/'setting.json', 'r') as f:
    args = json.load(f)
p                   = args['p']
r                   = args['yDim']
N                   = args['nTrain']
Nval                = args['nVal']
Ntest               = args['nTest']
model_pop_name      = args['model']
model_pop_TXrefMean = args['TXrefMean']
model_pop_Tstar     = args['Tstar']
# load pop_pattern
support_path_pop  = save(path_pop)
pop_pattern       = support_path_pop.load_dict('pattern')
# load pop_params and init
unscaled_pop_dict = support_path_pop.load_dict('noisy_unscaled_param')
pop_dict          = support_path_pop.load_dict('param') # only used in error computation and data generation

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

# sample data
if model_pop_name == 'HN':
    model_pop = HN()
elif model_pop_name == 'HP':
    model_pop = HP()
else:
    model_pop = HTP(Tstar=model_pop_Tstar)
ntot  = max(N)+Nval+Ntest

logger.info('start sampling')
try:
    np.random.seed(rep)
    torch.manual_seed(rep)
    torch.cuda.manual_seed(rep)
    YY        = (torch.rand(1,ntot)>.5)*2.-1.
    XX,HH,XXk = model_pop.sample_count(YY,pop_dict,pop_pattern,model_pop_TXrefMean,separate_sampling = True)
except Exception as ex:
    logger.warning(f"sampler broken with exception {str(ex)}")

# transform data 
def transform_data(XX, HH, XXk, model):
    if model.__class__.__name__ == 'BPGM':
        ZZ = HH
    elif model.__class__.__name__   == 'NPGM':
        ZZ            = torch.randn_like(XX)*1e-3
        ZZ[HH.bool()] = torch.log(XX/XXk)[HH.bool()]# use log transform
    elif model.__class__.__name__ == 'TPPGM':
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
    if model_name   == 'BPGM':
        model = BPGM(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
    elif model_name   == 'NPGM':
        model = NPGM(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
    elif model_name == 'PPGM':
        model = PPGM(batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
    else:
        if Tstar is None:
            Tstar=np.percentile(XX[:,:n][HH[:,:n].bool()].cpu().numpy(), Tstar_percentile)
        model = TPPGM(Tstar=Tstar,batch_size_jacobian=batch_size_jacobian,batch_size_hessian=batch_size_hessian)
    ############ transform data
    ZZ = transform_data(XX, HH, XXk, model)

    # split data
    Ytrain,Yval,Ytest = torch.split(YY,[max(N),Nval,Ntest],dim=1)
    Ztrain,Zval,Ztest = torch.split(ZZ,[max(N),Nval,Ntest],dim=1)
    Htrain,Hval,Htest = torch.split(HH,[max(N),Nval,Ntest],dim=1)
    # take the portion of data
    Z,H,Y         =Ztrain[:,:n],Htrain[:,:n],Ytrain[:,:n]
    
    # pop reduction in test to measure discrepancy
    Ztest_pop    = transform_data(XX[:,max(N)+Nval:],Htest,XXk[:,max(N)+Nval:],model_pop)
    JR_pop       = pop_dict['Gamma'].t().mm(Ztest_pop)+pop_dict['Psi'].t().mm(Htest)
    JR_pop      -=JR_pop.mean()
    JR_pop      /=JR_pop.std()
    AUC_TEST_POP = roc_auc_score(Ytest.squeeze().cpu().numpy(),JR_pop.squeeze().cpu().numpy())
    jr_pen       = lambda JR_hat : normalized_projection_error((JR_hat-JR_hat.mean(1,keepdims=True)).squeeze(), (JR_pop-JR_pop.mean(1,keepdims=True)).squeeze(), 1)
    jr_per       = lambda JR_hat : relative_projection_error((JR_hat-JR_hat.mean(1,keepdims=True)).squeeze(), (JR_pop-JR_pop.mean(1,keepdims=True)).squeeze())
    jr_sa        = lambda JR_hat : subespace_angle((JR_hat-JR_hat.mean(1,keepdims=True)).t(), (JR_pop-JR_pop.mean(1,keepdims=True)).t())
    JSPN_pop     = torch.cat((pop_dict['Gamma'],pop_dict['Psi']),dim                                        = 0)
    jspn_pen     = lambda JSPN_hat : normalized_projection_error(JSPN_hat.squeeze(), JSPN_pop.squeeze(), 1)
    jspn_per     = lambda JSPN_hat : relative_projection_error(JSPN_hat.squeeze(), JSPN_pop.squeeze(),)
    jspn_sa      = lambda JSPN_hat : subespace_angle(JSPN_hat, JSPN_pop)
    ############ train
    # train indep model
    if tensorboard:
        sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/'indep',**args_summary_writer)
    else:
        sw=None
    if model.__class__.__name__=='BPGM':
    	indep_dict = {'eta':pop_dict['xi'].clone()}#model.init_dict(Z,Y)
    else:
    	indep_dict = {k:v.clone() for k,v in pop_dict.items() if k in ('eta','ThetaDiag')}#model.init_dict(Z,Y)
    indep_dict = PGM_newton_indep_model(model=model,X=Z,Y=Y,indep_params=indep_dict,**args_newton,summary_writer=sw)
    if save_models:
        support_path_n_rep = save(path_n_rep)
        support_path_n_rep.save_dict(indep_dict,'indep_hat')
    indep_dict.update(dict(Gamma=torch.zeros(p,Y.size(0)),Theta=torch.zeros(p,p)))
    # construct fisher at independent
    PF  =PGM_hessian_fisher(model,param_dict=indep_dict, X=Z, Y=Y,**dict(fisher=True,burn_in=0,damping=damping,mineig=mineig,maxeig=1e99))

    if model.__class__.__name__=='BPGM':
    	init_dict={}
    	init_dict['eta']  =indep_dict['eta'].clone()#pop_dict['xi'].clone()
    	init_dict['Gamma']=unscaled_pop_dict['Psi'].clone()
    	init_dict['Theta']=unscaled_pop_dict['Lambda'].clone()
    else:
    	# init_dict = {k:v.clone() for k,v in pop_dict.items() if k in ('eta','Gamma','Theta','ThetaDiag')}
        init_dict={}
        for k in ['eta','ThetaDiag']:
            if k in indep_dict.keys():
                init_dict[k]=indep_dict[k].clone()#pop_dict[k].clone()
        for k in ['Gamma','Theta']:
            init_dict[k]=unscaled_pop_dict[k].clone()
        # scale params 
        if model.__class__.__name__ == 'NPGM':
            _iscale = (-1/indep_dict['ThetaDiag'].squeeze()).sqrt()
        elif model.__class__.__name__ in ['PPGM','TPPGM']:
            _iscale = torch.clamp_min(indep_dict['eta'].squeeze().exp(),1e-3).sqrt()
        else:
            _iscale = torch.ones_like(indep_dict['eta'].squeeze())
        init_dict['Theta']*=torch.ger(1/_iscale,1/_iscale)
        init_dict['Gamma']*=1/_iscale[:,None]      
        # project onto param space
        init_dict['Theta'].fill_diagonal_(0)
        init_dict['Theta']=(init_dict['Theta']+init_dict['Theta'].t())/2
        init_dict['Theta'],_=model.project_Theta(init_dict['Theta'],init_dict['ThetaDiag'] if 'ThetaDiag' in init_dict.keys() else None)
    
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

    # construct hessian
    time_init_PFH=time.process_time()
    PFH =PGM_hessian_fisher(model,param_dict=init_dict, X=Z, Y=Y,**dict(fisher=pseudoFisher,damping=damping,mineig=mineig,burn_in=1000,maxeig=1e99))
    time_elapsed_PFH = time.process_time() - time_init_PFH
    def train(lambda_reginter,lambda_inter,refit=True,summary_writer_pnopt=None,summary_writer_refit=None):
        # sw=torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'pnopt_{lambda_reginter:.6f}_{lambda_inter:.6f}')
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
            refitted_hat = PGM_pnopt_pattern(model=model,FisherObj=PFHR,
                                          init_dict=r_init_dict,pattern_dict=pnopt_hat_pattern,
                                          X=Z, Y=Y,
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
              'FPR_VS','FNR_VS','FPR_CI','FNR_CI','FPR_CI_SH','FNR_CI_SH',
              'AUC_TEST','AUC_TEST_pop','AUC_VAL',
              'ELL_VAL',
              'AIC','BIC',
              'JR_PEN','JR_PER','JR_SA',
              'JSPN_PEN','JSPN_PER','JSPN_SA']:
        results[k] = []
    for i_lambda_reginter, lambda_reginter in enumerate(np.linspace(lambda_reginter_max,0,num=gridsize)):
        for i_lambda_inter, lambda_inter in enumerate(np.linspace(lambda_inter_max,0,num=gridsize)):
            logger.info(f"start i_lambda_reginter{i_lambda_reginter}_i_lambda_inter{i_lambda_inter}") 
            try:
                if tensorboard:
                    sw =torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'pnopt_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                    swr=torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'pnopt_pattern_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                else:
                    sw=None
                    swr=None
                pattern_hat, pnopt_hat, refitted_hat, time_pnopt, time_refit = train(lambda_reginter,lambda_inter,refit=True,summary_writer_pnopt=sw,summary_writer_refit=swr)
                # pattern_hat, pnopt_hat, refitted_hat, time_pnopt, time_refit = train(lambda_reginter,lambda_inter,refit=False,summary_writer_pnopt=sw,summary_writer_refit=swr)
                if save_models:
                    support_path_n_rep = save(path_n_rep)
                    support_path_n_rep.save_dict(refitted_hat,f'refited_hat_{i_lambda_reginter}_{i_lambda_inter}')
                    support_path_n_rep.save_dict(pattern_hat,f'pattern_hat_{i_lambda_reginter}_{i_lambda_inter}')
                    support_path_n_rep.save_dict(pnopt_hat,f'pnopt_hat_{i_lambda_reginter}_{i_lambda_inter}')
                   
                # evaluate performance
                FPR_FNR_vs            = FPR_FNR(pattern_hat['vs'].squeeze(),pop_pattern['vs'].squeeze())
                FPR_FNR_ci            = FPR_FNR(pattern_hat['ci'],pop_pattern['ci'].squeeze())
                FPR_FNR_ci_SH         = FPR_FNR(pattern_hat['ci'][pop_pattern['vs'].squeeze(),:][:,pop_pattern['vs'].squeeze()],
                                                pop_pattern['ci'][pop_pattern['vs'].squeeze(),:][:,pop_pattern['vs'].squeeze()])
                # AUC in test
                Rtest                 = refitted_hat['Gamma'].t().mm(Ztest)#+refitted_hat['Psi'].t().mm(Htest)
                AUC_TEST              = roc_auc_score(Ytest.t().cpu().numpy(),Rtest.t().cpu().numpy())
                # evaluate criterions
                Rval                  = refitted_hat['Gamma'].t().mm(Zval)#+refitted_hat['Psi'].t().mm(Hval)
                AUC_VAL               = roc_auc_score(Yval.t().cpu().numpy(),Rval.t().cpu().numpy())
                ELL_VAL               = float(model.ell(Zval,Yval,refitted_hat).cpu().numpy())
                AIC,BIC,_             = model.compute_AIC_BIC(Z,Y,refitted_hat,pattern_hat,count_dim = True)
                # SPN
                JR_hat                = refitted_hat['Gamma'].t().mm(Ztest)#+refitted_hat['Psi'].t().mm(Htest)
                JR_hat               -=JR_hat.mean()
                JR_hat               /=JR_hat.std()
                
                JR_PEN                = jr_pen(JR_hat)
                JR_PER                = jr_per(JR_hat)
                JR_SA                 = jr_sa(JR_hat)
                if model.__class__.__name__=='BPGM':
                    JSPN_hat              = torch.cat((torch.zeros_like(refitted_hat['Gamma']),refitted_hat['Gamma']),dim = 0)
                else:
                    JSPN_hat              = torch.cat((refitted_hat['Gamma'],torch.zeros_like(refitted_hat['Gamma'])),dim = 0)
                JSPN_PEN              = jspn_pen(JSPN_hat)
                JSPN_PER              = jspn_per(JSPN_hat)
                JSPN_SA               = jspn_sa(JSPN_hat)
                # save some results
                if tensorboard:
                    swp =torch.utils.tensorboard.SummaryWriter(log_dir=path_n_rep/f'results_{i_lambda_reginter}_{i_lambda_inter}',**args_summary_writer)
                    swp.add_scalars('selection/vs',dict(zip(['FPR','FNR'],FPR_FNR_vs)))
                    swp.add_scalars('selection/ci',dict(zip(['FPR','FNR'],FPR_FNR_ci)))
                    swp.add_scalar('criterion/AUC_val', AUC_VAL)
                    swp.add_scalar('criterion/AIC', AIC)
                    swp.add_scalar('criterion/BIC', BIC)
                    swp.add_scalar('performance/AUC_TEST', AUC_TEST)
                    swp.add_scalar('performance/JR_PEN', JR_PEN)
                    swp.add_scalar('performance/JSPN_PEN', JSPN_PEN)
            except Exception as ex:
                logger.warning(f"train broken {str(ex)}")
                time_pnopt    = np.nan
                time_refit    = np.nan
                FPR_FNR_vs    = np.nan,np.nan
                FPR_FNR_ci    = np.nan,np.nan
                FPR_FNR_ci_SH = np.nan,np.nan
                # AUC in test
                AUC_TEST      = np.nan
                # evaluate cr
                AUC_VAL       = np.nan
                ELL_VAL       = np.nan
                AIC,BIC       = np.nan, np.nan
                # SPN
                JR_PEN        = np.nan
                JR_PER        = np.nan
                JR_SA         = np.nan
                JSPN_PEN      = np.nan
                JSPN_PER      = np.nan
                JSPN_SA       = np.nan
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
            results['FPR_CI'].append(FPR_FNR_ci[0])
            results['FNR_CI'].append(FPR_FNR_ci[1])
            results['FPR_CI_SH'].append(FPR_FNR_ci_SH[0])
            results['FNR_CI_SH'].append(FPR_FNR_ci_SH[1])
            results['AUC_TEST'].append(AUC_TEST)
            results['AUC_TEST_pop'].append(AUC_TEST_POP)
            results['AUC_VAL'].append(AUC_VAL)
            results['ELL_VAL'].append(ELL_VAL)
            results['AIC'].append(float(AIC))
            results['BIC'].append(float(BIC))
            results['JR_PEN'].append(JR_PEN)
            results['JR_PER'].append(JR_PER)
            results['JR_SA'].append(JR_SA)
            results['JSPN_PEN'].append(JSPN_PEN)
            results['JSPN_PER'].append(JSPN_PER)
            results['JSPN_SA'].append(JSPN_SA)
    with open(path_n_rep/('results.json'),'w') as f:
        json.dump(results,f)
    logger.info('end')
