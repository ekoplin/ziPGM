# -*- coding: utf-8 -*-
'''
this script:
-load partial results
-combine results
-select models
-plot results 
'''
import argparse
import pathlib
import json
from datetime import date

# from copy import deepcopy
# from functools import cmp_to_key

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import numpy as np
import re
regex_outcome     = re.compile('^binary|continous')
regex_predictors = re.compile('.*?_(\BX|H\B)')
parser = argparse.ArgumentParser(description='fast variable selection with hatY model')
parser.add_argument('paths', nargs='*', type =str, help ='multiple paths where find config files and nested resutls')# required
paths=parser.parse_args(['binary_outcome_signal_H_abundant',
                         'binary_outcome_signal_X_abundant',
                         'continous_outcome_signal_H_abundant',
                         'continous_outcome_signal_X_abundant'
                         ]).paths
DF=[]
PATH={}
for path_str in paths:
    pop_path = pathlib.Path(path_str)
    pop_name = regex_outcome.findall(pop_path.name)[0]
    pop_pred =regex_predictors.findall(pop_path.name)[0]
    PATH[(pop_pred,pop_name)]=pop_path
    # find hat subfolders
    for hat_path in pop_path.iterdir():
        if hat_path.is_file():
            continue
        hat_name = hat_path.name
        if hat_name not in ['HN','HP','HTP']:
            continue
        # find directories staring with n
        for n_path in hat_path.iterdir():
            if n_path.is_file() or n_path.name[0]!='n':
                continue
            # find repetitions
            for rep_path in n_path.iterdir():
                try:

                    df=pd.read_json(rep_path/'results.json')
                    df.insert(0,'pred',pop_pred)
                    # df.insert(1,'strength',pop_strength)
                    df.insert(2,'pop',pop_name)
                    df.insert(3,'hat',hat_name)
                    DF.append(df)
                except:
                    continue
df = pd.concat(DF)
df.set_index(['pred', 'pop', 'hat', 'n', 'rep', 'i_lambda_reginter', 'i_lambda_inter'],inplace=True)
df.dropna(axis=0,subset=['PREDERR_TEST', 'PREDERR_TEST_pop', 'PREDERR_VAL'],inplace=True)
df=df[df['FPR_VS']<1]
filtered=[]
for ig,g in df.reset_index().groupby(['pred', 'pop', 'hat', 'n']):
    valid_reps=g.groupby('rep').size().sort_values(ascending=False)[:100].index
    filtered.append(g[g['rep'].isin(valid_reps)])
df=pd.concat(filtered)
df.set_index(['pred', 'pop', 'hat', 'n', 'rep', 'lambda_reginter_max',
              'lambda_inter_max', 'lambda_reginter', 'lambda_inter',
              'i_lambda_reginter', 'i_lambda_inter'],inplace=True)
dfnew=df.query("pop=='binary' and PREDERR_VAL<0.5").apply(lambda x: 1-x if x.name=='PREDERR_VAL' else x)
df.loc[dfnew.index]=dfnew
dfnew=df.query("pop=='binary' and PREDERR_TEST<0.5").apply(lambda x: 1-x if x.name=='PREDERR_TEST' else x)
df.loc[dfnew.index]=dfnew
dfnew=df.query("pop=='binary' and PREDERR_TEST_pop<0.5").apply(lambda x: 1-x if x.name=='PREDERR_TEST_pop' else x)
df.loc[dfnew.index]=dfnew
# select
criterium_idx_binary = df.xs('binary',level=1).groupby(['pred', 'hat', 'n', 'rep'])['PREDERR_VAL'].idxmax()# AUC
binary_df=df.xs('binary',level=1).loc[criterium_idx_binary]
binary_df['criterion']='AUC'
criterium_idx_continous = df.xs('continous',level=1).groupby(['pred', 'hat', 'n', 'rep'])['PREDERR_VAL'].idxmin()# MSE
continous_df=df.xs('continous',level=1).loc[criterium_idx_continous]
continous_df['criterion']='MSE'
criterium_df=pd.concat({'binary':binary_df,'continous':continous_df},names=['pop'])
df_all=pd.concat([oracle,criterium_df.reorder_levels(order=oracle.index.names)])
hm_c=df_all.xs('continous',level=1).loc[:,('criterion','PREDERR_TEST','FPR_VS','FNR_VS')].rename(columns={'PREDERR_TEST':'MSE','FPR_VS':'FPR','FNR_VS':'FNR'}).droplevel([ 'lambda_reginter_max', 'lambda_inter_max', 'lambda_reginter', 'lambda_inter', 'i_lambda_reginter', 'i_lambda_inter'],axis=0).reset_index(level=1)
hm_b=df_all.xs('binary',level=1).loc[:,('criterion','PREDERR_TEST','FPR_VS','FNR_VS')].rename(columns={'PREDERR_TEST':'AUC','FPR_VS':'FPR','FNR_VS':'FNR'}).droplevel([ 'lambda_reginter_max', 'lambda_inter_max', 'lambda_reginter', 'lambda_inter', 'i_lambda_reginter', 'i_lambda_inter'],axis=0).reset_index(level=1)
# define estimator
hm_c['estimator']=hm_c['hat']#+'('+hm_c['criterion']+')'
hm_b['estimator']=hm_b['hat']#+'('+hm_b['criterion']+')'
# combine binary outcome cases
all_b=hm_b
all_b['outcome']='binary'
all_b.rename(columns={'AUC':'AUC/MSE'},inplace=True)
all_c=hm_c
all_c['outcome']='continous'
all_c.rename(columns={'MSE':'AUC/MSE'},inplace=True)
# combine all
all_=pd.concat([all_b,all_c])
# setting identifier
all_=all_.reset_index()
all_['setting']='outcome: '+all_['outcome'] +', predictors: '+all_['pred']
# population
pop_c=df_all.xs('continous',level=1).loc[:,('criterion','PREDERR_TEST_pop','FPR_VS','FNR_VS')].rename(columns={'PREDERR_TEST_pop':'MSE'}).droplevel([ 'lambda_reginter_max', 'lambda_inter_max', 'lambda_reginter', 'lambda_inter', 'i_lambda_reginter', 'i_lambda_inter'],axis=0)
pop_c=pop_c[pop_c['criterion']=='oracle']
pop_c['criterion']='pop'
pop_b=df_all.xs('binary',level=1).loc[:,('criterion','PREDERR_TEST_pop','FPR_VS','FNR_VS')].rename(columns={'PREDERR_TEST':'AUC'}).droplevel([ 'lambda_reginter_max', 'lambda_inter_max', 'lambda_reginter', 'lambda_inter', 'i_lambda_reginter', 'i_lambda_inter'],axis=0)
pop_b=pop_b[pop_b['criterion']=='oracle']
pop_b['criterion']='pop'
#%% load populational results
DF=[]
PATH={}
for path_str in paths:
    pop_path = pathlib.Path(path_str)
    pop_name = regex_outcome.findall(pop_path.name)[0]
    pop_pred =regex_predictors.findall(pop_path.name)[0]
    PATH[(pop_pred,pop_name)]=pop_path
    # find hat subfolders
    for hat_path in pop_path.iterdir():
        if hat_path.is_file():
            continue
        hat_name = hat_path.name
        if hat_name!='populational':
            continue
        # find directories staring with n
        for n_path in hat_path.iterdir():
            if n_path.is_file() or n_path.name[0]!='n':
                continue
            # find repetitions
            for rep_path in n_path.iterdir():
                try:

                    df=pd.read_json(rep_path/'results.json')
                    df.insert(0,'pred',pop_pred)
                    # df.insert(1,'strength',pop_strength)
                    df.insert(2,'pop',pop_name)
                    df.insert(3,'hat',hat_name)
                    DF.append(df)
                except:
                    continue
df = pd.concat(DF)
df.set_index(['pred', 'pop', 'hat', 'n', 'rep'],inplace=True)
pop_percentiles={}
for ig,g in df.groupby(['pred', 'pop', 'n']):
    pop_percentiles[ig]=np.nanpercentile(g, [25,50,75])
#%% rename
all_.rename(columns={'JR_PEN':'err(pred)',
                     'JR_SA':'sa(pred)',
                     'FNR_VS':'FNR(VS)','FPR_VS':'FPR(VS)',
                       },inplace=True)
df1=all_.copy()
df1=df1.reset_index()
df1=df1.replace({'HN':'Normal-zipGM',
               'HP':'Poisson-zipGM',
               'HTP':'TPoisson-zipGM',
               'PPGM':'Poisson-pGM',
               'Normal-pGM':'Normal-zipGM',
               'BPGM':'Ising-pGM'})
all_=df1#.set_index(all_.index.names)
palette=dict(zip(['Poisson-zipGM','Poisson-pGM','TPoisson-zipGM','TPoisson-pGM','Normal-zipGM','Normal-pGM','Ising-pGM','mixOmics'],[ c for ic,c in enumerate(sns.color_palette("Paired")) if ic in [0,1,2,3,4,5,7,10]]))
order=['Normal-zipGM','NPGM','Poisson-zipGM','PPGM','TPoisson-zipGM','TPPGM','BPGM']
order=['Normal-zipGM','Poisson-zipGM','TPoisson-zipGM']

#%% plots
sns.set(font_scale=1.2,style='white')
df_long=pd.melt(all_[all_.criterion!='oracle'].reset_index(),id_vars=['setting','estimator','outcome','n'],var_name='measure', 
                 value_vars=['AUC/MSE','FPR','FNR'],value_name='value')
df_long['n']=df_long['n'].astype(int)

for outcome,settings in [('binary',['outcome: binary, predictors: H','outcome: binary, predictors: X']),('continous',['outcome: continous, predictors: H','outcome: continous, predictors: X'])]:
    g=sns.catplot(data=df_long,x='n',y='value',hue='estimator',row='measure',col='setting',kind='box',sharey='row',showmeans=False,meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black","markersize":"5"},margin_titles=False,
                  palette=palette,col_order=settings,flierprops = dict(markerfacecolor = '0.50', markersize = 1),whis=1.5,showfliers = False,
                  hue_order=order)#,sharey='col'
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0), ncol=4, title='model', frameon=True)
    for predictor in ['X','H']:
        g.axes_dict[('AUC/MSE',f'outcome: {outcome}, predictors: {predictor}')].plot([pop_percentiles[(predictor,outcome,n)][0] for n in (200,500,1000)],color='k',linestyle =':')
        g.axes_dict[('AUC/MSE',f'outcome: {outcome}, predictors: {predictor}')].plot([pop_percentiles[(predictor,outcome,n)][1] for n in (200,500,1000)],color='k',linestyle =':')
        g.axes_dict[('AUC/MSE',f'outcome: {outcome}, predictors: {predictor}')].plot([pop_percentiles[(predictor,outcome,n)][2] for n in (200,500,1000)],color='k',linestyle =':')
    
    for ax in list(g.axes_dict.keys()):
        if ax[1][-1]=='H':
            if ax[0]=='AUC/MSE':
                g.axes_dict[ax].set_ylabel('AUC' if outcome=='binary' else 'MSE')
            else:
                g.axes_dict[ax].set_ylabel(f"{ax[0]}(VS)")
        else:
            g.axes_dict[ax].set_ylabel('')
        if ax[0]=='AUC/MSE':
            # g.axes_dict[ax].set_title(f'{ax[1].split(", ")[1]}')
            if ax[1].split(", ")[1]=='predictors: X':
                g.axes_dict[ax].set_title('Y|X')
            else:
                g.axes_dict[ax].set_title(r'$Y|\nu(X)$')
        else:
            g.axes_dict[ax].set_title('')
    
    g.savefig(f'comparisions_prederr_{outcome}.eps')
    # oracle
    df_long=pd.melt(all_[all_.criterion=='oracle'].reset_index(),id_vars=['setting','estimator','outcome','n'],var_name='measure', 
                     value_vars=['AUC/MSE','FPR','FNR'],value_name='value')
    df_long['n']=df_long['n'].astype(int)
    g=sns.catplot(data=df_long,x='n',y='value',hue='estimator',row='measure',col='setting',kind='box',sharey='row',showmeans=False,meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black","markersize":"5"},margin_titles=False,
                  palette=palette,col_order=settings,flierprops = dict(markerfacecolor = '0.50', markersize = 1),whis=1.5,showfliers = False,
                  hue_order=order)
    g.set_ylabels('measure')
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0), ncol=4, title='model', frameon=True)
    for predictor in ['X','H']:
        g.axes_dict[('AUC/MSE',f'outcome: {outcome}, predictors: {predictor}')].plot([pop_percentiles[(predictor,outcome,n)][0] for n in (200,500,1000)],color='k',linestyle =':')
        g.axes_dict[('AUC/MSE',f'outcome: {outcome}, predictors: {predictor}')].plot([pop_percentiles[(predictor,outcome,n)][1] for n in (200,500,1000)],color='k',linestyle =':')
        g.axes_dict[('AUC/MSE',f'outcome: {outcome}, predictors: {predictor}')].plot([pop_percentiles[(predictor,outcome,n)][2] for n in (200,500,1000)],color='k',linestyle =':')
    
    for ax in list(g.axes_dict.keys()):
        if ax[1][-1]=='H':
            if ax[0]=='AUC/MSE':
                g.axes_dict[ax].set_ylabel('AUC' if outcome=='binary' else 'MSE')
            else:
                g.axes_dict[ax].set_ylabel(f"{ax[0]}(VS)")
        else:
            g.axes_dict[ax].set_ylabel('')
        if ax[0]=='AUC/MSE':
            g.axes_dict[ax].set_title(f'{ax[1].split(", ")[1]}')
        else:
            g.axes_dict[ax].set_title('')
    g.savefig(f'comparisions_oracle_{outcome}.eps')
