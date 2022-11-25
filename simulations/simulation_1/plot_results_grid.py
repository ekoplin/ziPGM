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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import numpy as np
import re
regex_name     = re.compile('\BHTP|HP|HN\B')
regex_structure = re.compile('(^.*?-.*?-.*?)\-')
regex_strength = re.compile('^.*?-.*?-.*?\-(.*?)_')
regex_dxi = re.compile('^.*dxi(\d)')


parser = argparse.ArgumentParser(description='fast variable selection with hatY model')
parser.add_argument('paths', nargs='*', type =str, help ='multiple paths where find config files and nested resutls')# required
paths=parser.parse_args([ 'blockdiag-indep-indep-0.0050_0.1200_0.3800_popHP_binary1_p100_dxi7',#
                          'blockdiag-indep-indep-0.0050_0.1200_0.3800_popHP_binary1_p100_dxi5',#
                          'blockdiag-indep-indep-0.0050_0.1200_0.3800_popHP_binary1_p100_dxi3',#
                          'blockdiag-indep-indep-0.0050_0.0850_0.3100_popHTP_binary1_p100_dxi7',#
                          'blockdiag-indep-indep-0.0050_0.0850_0.3100_popHTP_binary1_p100_dxi5',#
                          'blockdiag-indep-indep-0.0050_0.0850_0.3100_popHTP_binary1_p100_dxi3',#
                          'blockdiag-indep-indep-0.0100_0.0700_0.1500_popHN_binary1_p100_dxi7',#
                          'blockdiag-indep-indep-0.0100_0.0700_0.1500_popHN_binary1_p100_dxi5',#
                          'blockdiag-indep-indep-0.0100_0.0700_0.1500_popHN_binary1_p100_dxi3',#
                          ]).paths
DF=[]
PATH={}
for path_str in paths:
    pop_path = pathlib.Path(path_str)
    pop_name = regex_name.findall(pop_path.name)[0]
    pop_structure=regex_structure.findall(pop_path.name)[0]
    pop_strength =regex_strength.findall(pop_path.name)[0]
    pop_dxi      =regex_dxi.findall(pop_path.name)[0]
    PATH[(pop_structure,pop_strength,pop_name)]=pop_path
    # find hat subfolders
    for hat_path in pop_path.iterdir():
        if hat_path.is_file():
            continue
        hat_name = hat_path.name
        if hat_name not in ['HP','HTP','HN','PPGM','TPPGM','BPGM','NPGM']:
            continue
        # find directories staring with n
        for n_path in hat_path.iterdir():
            if n_path.is_file() or n_path.name[0]!='n':
                continue
            # find repetitions
            for rep_path in n_path.iterdir():
                try:

                    df=pd.read_json(rep_path/'results.json')
                    df.insert(0,'ci',pop_structure)
                    df.insert(1,'strength',pop_strength)
                    df.insert(2,'dxi',pop_dxi)
                    df.insert(3,'pop',pop_name)
                    df.insert(4,'hat',hat_name)
                    DF.append(df)
                except:
                    continue
df = pd.concat(DF)
df.set_index(['ci', 'strength','dxi', 'pop', 'hat', 'n', 'rep', 'lambda_reginter_max',
              'lambda_inter_max', 'lambda_reginter', 'lambda_inter',
              'i_lambda_reginter', 'i_lambda_inter'],inplace=True)
df.dropna(axis=0,subset=['AUC_TEST', 'AUC_TEST_pop', 'AUC_VAL'],inplace=True)
df=df[df['FPR_VS']<1]
filtered=[]
for ig,g in df.reset_index().groupby(['ci', 'strength', 'pop', 'hat', 'n']):
    valid_reps=g.groupby('rep').size().sort_values(ascending=False)[:100].index
    filtered.append(g[g['rep'].isin(valid_reps)])
df=pd.concat(filtered)
df.set_index(['ci', 'strength', 'pop', 'hat', 'n', 'rep', 'lambda_reginter_max',
              'lambda_inter_max', 'lambda_reginter', 'lambda_inter',
              'i_lambda_reginter', 'i_lambda_inter'],inplace=True)
df[df['AUC_VAL']<0.5]=df[df['AUC_VAL']<0.5].apply(lambda x: 1-x if x.name=='AUC_VAL' else x)
df[df['AUC_TEST']<0.5]=df[df['AUC_TEST']<0.5].apply(lambda x: 1-x if x.name=='AUC_TEST' else x)
df[df['AUC_TEST_pop']<0.5]=df[df['AUC_TEST_pop']<0.5].apply(lambda x: 1-x if x.name=='AUC_TEST_pop' else x)
#%%
oracle_valid= df[df['FPR_VS']<=.1]
oracle_idx      = oracle_valid.groupby(['ci', 'strength','dxi', 'pop', 'hat', 'n', 'rep'])['FNR_VS'].idxmin()
oracle = df.loc[oracle_idx]
oracle['criterion']='oracle'
criterium_idx_binary = df.groupby(['ci', 'strength','dxi', 'pop', 'hat', 'n', 'rep'])['AUC_VAL'].idxmax()# AUC
binary_df=df.loc[criterium_idx_binary]
binary_df['criterion']='AUC'
df_all=pd.concat([oracle,binary_df.reorder_levels(order=oracle.index.names)])
#%% rename
df_all.rename(columns={'AUC_TEST':'AUC',
                       'JSPN_PEN':'err','JR_PEN':'err(pred)',
                       'JSPN_SA':'sa'  ,'JR_SA':'sa(pred)',
                       'FNR_VS':'FNR(VS)','FPR_VS':'FPR(VS)',
                       'FNR_CI':'FNR(CI)','FPR_CI':'FPR(CI)',
                       },inplace=True)
df1=df_all.copy()
df1=df1.reset_index()
df1=df1.replace({'HN':'Normal-zipGM',
               'HP':'Poisson-zipGM',
               'HTP':'TPoisson-zipGM',
               'BPGM':'Ising-pGM',
               'NPGM':'Normal-pGM',
               'PPGM':'Poisson-pGM',
               'TPPGM':'TPoisson-pGM'})

df1=df1.replace({'3':r'$\Delta_\xi=-2$',
               '5':r'$\Delta_\xi=0$',
               '7':r'$\Delta_\xi=2$'})
df_all=df1.set_index(df_all.index.names)
# plot
palette=dict(zip(['Poisson-zipGM','Poisson-pGM','TPoisson-zipGM','TPoisson-pGM','Normal-zipGM','Normal-pGM','Ising-pGM'],[ c for ic,c in enumerate(sns.color_palette("Paired")) if ic in [0,1,2,3,4,5,7]]))
#%% plot ALL
sns.set(font_scale=1.4,style='white')
for ig,g in df_all.groupby('pop'):
    print(ig)
    gg=g.reset_index()
    gg['setting']=gg['ci']+'('+gg['strength']+')'
    
    gg_auc=gg[gg.criterion!='oracle']
    gg_oracle=gg[gg.criterion=='oracle']
    
    for gg_,name in [(gg_auc,'prederr'),(gg_oracle,'oracle')]:
        if name=='oracle':
            df_long=pd.melt(gg_,id_vars=['dxi','hat','n'],var_name='measure', 
                         value_vars=['AUC','FPR(VS)','FNR(VS)'],value_name='value')
        else:
            df_long=pd.melt(gg_,id_vars=['dxi','hat','n'],var_name='measure', 
                         value_vars=['AUC','FPR(VS)','FNR(VS)','FPR(CI)','FNR(CI)'],value_name='value')
        g_main=sns.catplot(data=df_long,x='n',y='value',hue='hat',row='measure',col='dxi',kind='box',sharey='row',showmeans=False,meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black","markersize":"5"},margin_titles=False,
                      palette=palette,flierprops = dict(markerfacecolor = '0.50', markersize = 1),whis=1.5,showfliers = False)#,sharey='col'
        for g,gn in [(g_main,'main')]:#,(g_app,'appendix')]:
            g.set_ylabels('measure')
            sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0), ncol=5, title='model', frameon=True)
            for igs,gs in gg_oracle.groupby(['setting','dxi']):
                g.axes_dict[('AUC',igs[1])].plot([gs[(gs['pop']==gs['hat'])&(gs['n']==1000)]['AUC_TEST_pop'].quantile(q=.25)]*3,color='k',linestyle =':')
                g.axes_dict[('AUC',igs[1])].plot([gs[(gs['pop']==gs['hat'])&(gs['n']==1000)]['AUC_TEST_pop'].quantile(q=.5)]*3,color='k',linestyle =':')
                g.axes_dict[('AUC',igs[1])].plot([gs[(gs['pop']==gs['hat'])&(gs['n']==1000)]['AUC_TEST_pop'].quantile(q=.75)]*3,color='k',linestyle =':')
            for ax in list(g.axes_dict.keys()):
                g.axes_dict[ax].set_ylabel(f'{ax[0]}')
                if ax[0]=='AUC':
                    g.axes_dict[ax].set_title(f'{ax[1]}')
                else:
                    g.axes_dict[ax].set_title('')
            g.savefig(f'pop{ig}_{name}.eps')
