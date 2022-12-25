# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

gpt2_responses = pd.read_csv("gpt2_small_to_xl_responses.csv")
opt_responses = pd.read_csv("opt125m_to_30b_responses.csv")
gpt3_responses = pd.read_csv("gpt3_responses.csv")
human_responses = pd.read_csv("scope_experiment_c_and_c_results.csv")

def get_salient_model_data(df):
    responsecols = [x for x in df.columns if "response" in x]
    modelnames = [x.replace("_response","") for x in responsecols]
    
    SF1s = []
    SF2s = []
    ScF1s = []
    ScF2s = []
    
    for i in range(1,int(np.max(df["idx"])+1)):
        df_indexed = df[df["idx"]==i]
        SF1 = df_indexed[(df_indexed['stype']=='S') & (df_indexed['ftype']=='F1')][responsecols].values[0,:]
        SF2 = df_indexed[(df_indexed['stype']=='S') & (df_indexed['ftype']=='F2')][responsecols].values[0,:]
        ScF1 = df_indexed[(df_indexed['stype']=='Sc') & (df_indexed['ftype']=='F1')][responsecols].values[0,:]
        ScF2 = df_indexed[(df_indexed['stype']=='Sc') & (df_indexed['ftype']=='F2')][responsecols].values[0,:]
        
        SF1s.append(SF1)
        SF2s.append(SF2)
        ScF1s.append(ScF1)
        ScF2s.append(ScF2)
    
    SF1s = np.array(SF1s)
    SF2s = np.array(SF2s)
    ScF1s = np.array(ScF1s)
    ScF2s = np.array(ScF2s)
    
    S_diffs = SF1s - SF2s
    S_diffs_df = pd.DataFrame(S_diffs,columns=modelnames)
                
    Sc_diffs = ScF1s - ScF2s
    Sc_diffs_df = pd.DataFrame(Sc_diffs, columns=modelnames)
    
    
    pttest = stats.ttest_rel(S_diffs, Sc_diffs)
    
    diff_diffs = S_diffs - Sc_diffs
    diff_diffs = diff_diffs/np.std(diff_diffs,axis=0)
    diff_diffs_df = pd.DataFrame(diff_diffs, columns=modelnames)
    
    
    df_data = {'S_diffs': S_diffs_df, 'Sc_diffs': Sc_diffs_df,
               'diff_diffs': diff_diffs_df, 'pttest':pttest            
               }
    return df_data
    
gpt2_data = get_salient_model_data(gpt2_responses)    
opt_data = get_salient_model_data(opt_responses)
gpt3_data = get_salient_model_data(gpt3_responses)  

def get_salient_human_data(df, epsilon=0.01):
    
    SF1_vals = []
    SF2_vals = []
    ScF1_vals = []
    ScF2_vals = []
    OP1_types = []
    OP2_types = []
    OP1s = []
    OP2s = []
    indices = []
    
    for i in range(1,int(np.max(df["idx"])+1)):
        df_indexed = df[df["idx"]==i]
        SF1 = df_indexed[(df_indexed['stype']=='S') & (df_indexed['ftype']=='F1')]
        SF2 = df_indexed[(df_indexed['stype']=='S') & (df_indexed['ftype']=='F2')]
        ScF1 = df_indexed[(df_indexed['stype']=='Sc') & (df_indexed['ftype']=='F1')]
        ScF2 = df_indexed[(df_indexed['stype']=='Sc') & (df_indexed['ftype']=='F2')]
        
        SF1_val = np.log((np.mean(SF1['response'])-1+epsilon)/(6+epsilon))
        SF2_val = np.log((np.mean(SF2['response'])-1+epsilon)/(6+epsilon))
        ScF1_val = np.log((np.mean(ScF1['response'])-1+epsilon)/(6+epsilon))
        ScF2_val = np.log((np.mean(ScF2['response'])-1+epsilon)/(6+epsilon))
        
        SF1_vals.append(SF1_val)
        SF2_vals.append(SF2_val)
        ScF1_vals.append(ScF1_val)
        ScF2_vals.append(ScF2_val)
        
        OP1_type = list(df_indexed[df_indexed['stype']=='S']['OP1_type'])[0]
        OP2_type = list(df_indexed[df_indexed['stype']=='S']['OP2_type'])[0]
        OP1 = list(df_indexed[df_indexed['stype']=='S']['OP1'])[0]
        OP2 = list(df_indexed[df_indexed['stype']=='S']['OP2'])[0]
        
        OP1_types.append(OP1_type)
        OP2_types.append(OP2_type)
        OP1s.append(OP1)
        OP2s.append(OP2)
        
        indices.append(i)       
        

                
    SF1_vals = np.array(SF1_vals)
    SF2_vals = np.array(SF2_vals)
    ScF1_vals = np.array(ScF1_vals)
    ScF2_vals = np.array(ScF2_vals)
    
    S_diffs = SF1_vals-SF2_vals
    S_diffs_df = pd.DataFrame(S_diffs, columns=['human'])
    Sc_diffs = ScF1_vals-ScF2_vals
    Sc_diffs_df = pd.DataFrame(Sc_diffs,columns=['human'])
    
    pttest = stats.ttest_rel(S_diffs, Sc_diffs)
    
    diff_diffs = S_diffs - Sc_diffs
    diff_diffs = diff_diffs/np.std(diff_diffs,axis=0)
    diff_diffs_df = pd.DataFrame(diff_diffs, columns=['human'])
    
    
    operators_and_indices_df = pd.DataFrame(zip(indices, OP1_types,OP2_types,OP1s,OP2s),columns=['idx','OP1_type','OP2_type','OP1','OP2'])
    
    df_data = {'S_diffs': S_diffs_df, 'Sc_diffs': Sc_diffs_df,
               'diff_diffs': diff_diffs_df, 'pttest':pttest,
               'operators_and_indices': operators_and_indices_df
               }
    
    return df_data
        

human_data = get_salient_human_data(human_responses)

agg_data = pd.concat([human_data['operators_and_indices'], human_data['diff_diffs'], gpt2_data['diff_diffs'], opt_data['diff_diffs'], gpt3_data['diff_diffs']],axis=1)


modelnames = ['gpt2small','gpt2med', 'gpt2large','gpt2xl','opt125m', 'opt350m', 'opt1b', 'opt3b', 'opt7b', 'opt13b', 'opt30b', 'gpt3']


plt.figure()
ax = sns.histplot(agg_data['human'], bins=np.arange(-4,4,0.5))
ax.set_ylim(0,25)
ax.set_xlabel('Difference of log ratios of rating-derived scores,\n human baselines (standardized by st. dev.)')
for name in modelnames:
    plt.figure()
    ax = sns.histplot(agg_data[name], bins=np.arange(-4,4,0.5))
    ax.set_ylim(0,25)
    ax.set_xlabel(f'Difference of log ratios of probabilities,\n{name} (standardized by st. dev.)')


for name in modelnames:
    plt.figure()
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    fig.suptitle(name.upper(),fontsize=36)
    sns.scatterplot(ax=axes[0],data=agg_data,x=name,y='human',hue='OP1_type')
    axes[0].set(ylabel='Difference of log ratios of\n rating-derived scores, human baselines \n(standardized by st. dev.)', xlabel=f'Difference of log ratios of probabilities,\n{name} (standardized by st. dev.)')
    sns.scatterplot(ax=axes[1],data=agg_data,x=name,y='human',hue='OP2_type')
    axes[1].set(ylabel='Difference of log ratios of\n rating-derived scores, human baselines \n(standardized by st. dev.)', xlabel=f'Difference of log ratios of probabilities,\n{name} (standardized by st. dev.)')
    sns.regplot(ax=axes[2],data=agg_data,x=name,y='human')
    axes[2].set(ylabel='Difference of log ratios of\n rating-derived scores, human baselines \n(standardized by st. dev.)', xlabel=f'Difference of log ratios of probabilities,\n{name} (standardized by st. dev.)')
    linregress = stats.linregress(agg_data[name], agg_data['human'])
    print(f'{name.upper()}:\nSlope: {linregress[0]}\nIntercept: {linregress[1]}\nR-Value: {linregress[2]}\n')
    

agg_data_qdonly = agg_data[agg_data['OP2_type']=='QD']
for name in modelnames:
    plt.figure()
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    fig.suptitle(name.upper(),fontsize=36)
    sns.scatterplot(ax=axes[0],data=agg_data_qdonly,x=name,y='human',hue='OP1_type')
    axes[0].set(ylabel='Difference of log ratios of\n rating-derived scores, human baselines \n(standardized by st. dev.)', xlabel=f'Difference of log ratios of probabilities,\n{name} (standardized by st. dev.)')
    sns.scatterplot(ax=axes[1],data=agg_data_qdonly,x=name,y='human',hue='OP2_type')
    axes[1].set(ylabel='Difference of log ratios of\n rating-derived scores, human baselines \n(standardized by st. dev.)', xlabel=f'Difference of log ratios of probabilities,\n{name} (standardized by st. dev.)')
    sns.regplot(ax=axes[2],data=agg_data_qdonly,x=name,y='human')
    axes[2].set(ylabel='Difference of log ratios of\n rating-derived scores, human baselines \n(standardized by st. dev.)', xlabel=f'Difference of log ratios of probabilities,\n{name} (standardized by st. dev.)')
    linregress = stats.linregress(agg_data_qdonly[name], agg_data_qdonly['human'])
    print(f'{name.upper()}:\nSlope: {linregress[0]}\nIntercept: {linregress[1]}\nR-Value: {linregress[2]}')
    

