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
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import matplotlib.gridspec

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import numpy as np
import re

parser = argparse.ArgumentParser(description='fast variable selection with hatY model')
parser.add_argument('paths', nargs='*', type =str, help ='multiple paths where find config files and nested resutls')# required
paths=parser.parse_args(['./']).paths

DF=[]
PATH={}
for path_str in paths:
    pop_path = pathlib.Path(path_str)
    pop_name='SEX'
    PATH[pop_name]=pop_path
    
    
    data=pd.read_csv(pop_path/'HGP_SEX.csv',header=[0,1],index_col=0)
    taxa_names=data['Taxa'].columns
    # find hat subfolders
    for hat_path in pop_path.iterdir():
        if hat_path.is_file():
            continue
        hat_name = hat_path.name
        if hat_name not in ['HN','HP','HTP','PPGM','TPPGM','NPGM','BPGM']:
            continue
        # find directories staring with n
        for fold_path in hat_path.iterdir():
            if fold_path.is_file():
                continue
            # find repetitions
            for ifold_path in fold_path.iterdir():
                try:

                    df=pd.read_json(ifold_path/'results.json')
                    if hat_name  in ['HN','HP','HTP']:
                        df_a=df[['fold', 'ifold', 'lambda_reginter_max', 'lambda_inter_max',
                               'lambda_reginter', 'lambda_inter', 'i_lambda_reginter',
                               'i_lambda_inter', 'time_pnopt', 'time_refit', 'PREDERR_TEST_S']]
                        df_a=df_a.rename({'PREDERR_TEST_S':'PREDERR_TEST'},axis=1)
                    else:
                        df_a=df[['fold', 'ifold', 'lambda_reginter_max', 'lambda_inter_max',
                               'lambda_reginter', 'lambda_inter', 'i_lambda_reginter',
                               'i_lambda_inter', 'time_pnopt', 'time_refit', 'PREDERR_TEST']]
                    df_b=pd.DataFrame.from_records(df['SELECTED_VARS'])
                    df_b.columns=taxa_names
                    
                    df_a.insert(0,'pop',pop_name)
                    df_a.insert(1,'hat',hat_name)
                    
                    df=pd.concat([df_a,df_b],axis=1,keys=['index','taxa'],names=['part'])
                    # df=df_a
                    DF.append(df)
                except:
                    continue
df = pd.concat(DF)
df.loc[df[('index','PREDERR_TEST')]<.5,('index','PREDERR_TEST')]=1-df.loc[df[('index','PREDERR_TEST')]<.5,('index','PREDERR_TEST')]
df.dropna(axis=0,subset=[('index','PREDERR_TEST')],inplace=True)
filtered=[]
for ig,g in df.groupby([('index',a) for a in['pop', 'hat','fold']]):
    i_to_remove=g[g.taxa.sum(1)>15|g.index.isna()|g.index.isnull()][[('index','i_lambda_reginter'),('index','i_lambda_inter')]]
    filtered.append(g[~(g[('index','i_lambda_reginter')].isin(i_to_remove[('index','i_lambda_reginter')])&g[('index','i_lambda_inter')].isin(i_to_remove[('index','i_lambda_inter')]))])
df=pd.concat(filtered)
df.columns=df.columns.droplevel(0)
# find the ilamndas that optimize across inner folds
df_b=df.reset_index()
df_b_outer=df_b[df_b['ifold']==-1]
df_b_inner=df_b[df_b['ifold']!=-1]
indx=df_b_inner.set_index(['fold',  'hat','i_lambda_reginter', 'i_lambda_inter']).groupby(['fold', 'hat','i_lambda_reginter', 'i_lambda_inter'])['PREDERR_TEST'].mean().groupby(['fold', 'hat']).idxmax(skipna=True)
df_selected=df_b_outer.set_index(['fold', 'hat','i_lambda_reginter', 'i_lambda_inter']).loc[indx]
df_selected.rename(columns={'PREDERR_TEST':'AUC'},inplace=True)
df = pd.concat([df_selected.xs('HP',level=1).reset_index(drop=True),
                df_selected.xs('HTP',level=1).reset_index(drop=True),
                df_selected.xs('HN',level=1).reset_index(drop=True),
                df_selected.xs('PPGM',level=1).reset_index(drop=True),
                df_selected.xs('TPPGM',level=1).reset_index(drop=True),
                df_selected.xs('NPGM',level=1).reset_index(drop=True)],
               keys=['Poisson-zipGM',
                     'TPoisson-zipGM',
                     'Normal-zipGM',
                     'Poisson-pGM',
                     'TPoisson-pGM',
                     'Normal-pGM',
                     ],
               names=['model','fold'],join='inner')
#plots
palette=dict(zip(['Poisson-zipGM','Poisson-pGM','TPoisson-zipGM','TPoisson-pGM','Normal-zipGM','Normal-pGM','BPGM'],[ c for ic,c in enumerate(sns.color_palette("Paired")) if ic in [0,1,2,3,4,5,7]]))
sns.set(font_scale=1.2,style='white')
df_vs=df.drop(['AUC'],axis=1).groupby('model').sum(0)
cols=df_vs.columns
regex_filter     = re.compile("\bk__Bacteria;\b")
df_vs.columns=pd.Index([re.sub('k__Bacteria;',"",col,0) for col in cols])
#clustermap figure
g = sns.clustermap(df_vs.loc[:,(df_vs>0).any(0)].astype('float').T,figsize=(10,20),metric='hamming',yticklabels=True,
                    # cmap='gist_gray_r',vmin=0, vmax=5,
                    cmap=sns.color_palette("Greys",6),vmin=-.5, vmax=5.5,
                   cbar_kws={"orientation": "horizontal",'label': 'number of times the variable was selected'},cbar_pos=(.94, 0.09,.5,0.02))
colorbar = g.ax_heatmap.collections[0].colorbar
colorbar.set_ticks([0,1,2,3,4,5])

g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=30) 
g.savefig(pop_path/'HGP_SEX_variable_selection.eps')
#boxplot figure
g1=sns.boxplot(data=df['AUC'].reindex(df.index.get_level_values(0).unique()[g.dendrogram_col.reordered_ind],level=0).reset_index(),x='model',y='AUC',palette=palette,flierprops = dict(markerfacecolor = '0.50', markersize = 1),whis=1.5,showfliers = False)
plt.setp(g1.get_xticklabels(), rotation=30) 
plt.tight_layout()
g1.figure.savefig(pop_path/'HGP_SEX_AUC.eps')
