# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:51:59 2023

@author: MayaYanko
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn import metrics
import matplotlib.pyplot as plt
from rap_response_predictor import RapResponsePredictor
from config import *
from rap_model_analysis import plot_kaplan_meier, cox_model_summary, plot_roc_curve, r_squared, pred_vs_obs
from rap_model_analysis import plot_roc_curve
from sklearn.model_selection import train_test_split
from plotting_wrappers_ori import *


#%%

def read_data(data_file, index='SubjectId'):
    # read datafile
    if '.csv' in data_file:
        return pd.read_csv(data_file, low_memory=False, index_col=index)  
    elif '.xlsx' in data_file:
        return pd.read_excel(data_file, index_col=index)

def get_params(df, configuration_file,protein_filt=None):
    # read configuration file
    with open(configuration_file,'r') as f:
        conf = json.load(f)['rap_response_predictor']
    response_metric = conf['response_metrics'][0]
    params = conf[f'{response_metric}_initialization_params']
    if pd.isna(protein_filt):
        protein_filt = df.columns[START_COL:].tolist()
    params['proteins'] = list(df.columns[df.columns.isin(protein_filt)])

    return params,response_metric    


def data_prepare_random_split(df, configuration_file, seed=42, train_ratio = 0.75):
    # parse the data
    params,_ = get_params(df, configuration_file)
    
    df = df.rename({RESPONSE_METRIC:'Y'},axis=1).dropna(subset='Y')
    df['Y'] = df['Y'].apply(int)

    #dev-val split
    if train_ratio!=1:
        [dev_df, validation_df] = train_test_split(df, train_size=train_ratio, stratify=1-df[RESPONSE_METRIC], random_state=seed)
    else:
        dev_df=df
        validation_df=pd.DataFrame(index=[],columns=dev_df.columns)
    
    return dev_df, validation_df, params


def model_generator(df, response_metric, model_name, seed, params):
    
    np.random.seed(seed)
    rap_response_predictor = RapResponsePredictor(**params)
    rap_response_predictor.fit(df)
    output_file = f'./models/{model_name}.pkl'
    print(response_metric, rap_response_predictor.sp_auc)
    with open(output_file, 'wb') as f:
        pickle.dump(rap_response_predictor, f)
    raps = pd.DataFrame.from_dict({'SeqId':rap_response_predictor.RAPs.keys(),'Count':rap_response_predictor.RAPs.values()})
    raps.to_csv(f'./raps/{model_name}.csv')
    return rap_response_predictor.sp_auc
    
def model_predict(model_name, response_metric, df, flag_roc=1, flag_kaplan=1,flag_dev=0,flag_label_from_dev=0,flag_r2=0,title=''): 
    output_file = f'./models/{model_name}.pkl'
    with open(output_file, 'rb') as f:
        model = pickle.load(f)
    if flag_dev:
        mod_pred_obj = model.prediction
        val_pred = model.prediction.y_pred_sp
        val_y = model.prediction.y if flag_label_from_dev else df.rename({'Y':'y'},axis=1).y
        val_pred = val_pred[val_y.index]
    else:
        mod_pred_obj = model.predict(df)
        val_pred = mod_pred_obj.y_pred_sp #model.predict(df).y_pred_sp#_scaled
        val_y = df.Y
    fpr, tpr, thresholds = metrics.roc_curve(val_y, val_pred)
    auc=metrics.auc(fpr, tpr)
    print(response_metric, "sp", auc)
    #th0 = thresholds[np.argmax(tpr-fpr)]
    th0 = model.prediction.y_pred_sp.median()
    print('threshold:',th0)
    error_rate = (val_y-val_pred.map(lambda num: 1 if num>=th0 else 0)).abs().mean()
    print('Error rate:',error_rate)
    pval=0
    if flag_roc:
        plt.figure()
        pval = plot_roc_curve(val_y, val_pred, show_pval=True)
        plt.title(title)
        plt.show()
    hr=0
    if flag_kaplan:
        plt.figure()
        df['OS Event'] = df['OS Event'].fillna(0)
        cox = cox_model_summary(df, duration='OS Duration', event='OS Event', group='Y')
        hr = plot_kaplan_meier(data=df, duration='OS Duration', event='OS Event', group='Y')  # plot_kaplan_meir(model, df, val_pred,th)
        plt.title(f"val set\nHR={np.round(cox['hr'],2)}, (CI: {np.round(cox['hr_ci'][0],2)}-{np.round(cox['hr_ci'][1],2)}), p-value<0.001")
        plt.show()
    if flag_r2:
        pred_to_r2 = mod_pred_obj.y_pred_sp_scaled
        pred_vs_obs(pred={'AE in 6 months' : pred_to_r2.to_list()}, actual={'AE in 6 months' : val_y.to_list()}, interval=.1, r_square_fit=True, show_pval=True, aparametric_pval=False,
                             aparametric_pval_iter=20001,
                             response_metric_legend={'AE in 6 months' : f'{response_metric} in 180 days'},
                             plot=True)
    return auc, model, val_pred, val_y,hr, pval, error_rate

def string_split_handler(stri):
    max_line_size=10
    
    if type(stri)!=str or len(stri)<=max_line_size:
        return stri
    
    stri_split = stri.split()
    current_count=-1
    str_out=''
    for word in stri_split:
        if current_count<=0 and len(word)>=max_line_size:
            str_out=str_out+word+'\n'
            current_count=-1    
        elif current_count+len(word)+1<=max_line_size:
            current_count=current_count+len(word)+1
            str_out=str_out+' '+word
        else:
            str_out=str_out+'\n'+word
            current_count=len(word) 
    return str_out


def summary_plots(results_path,dev_auc_df=None,val_auc_df=None,x_metric='',y_metric='AUC',read_from_file=False,df_val_name='Val Set',df_dev_name='Dev Set'):
    #x_metric=values_title
    #summary_plots(results_path,x_metric=values_title,read_from_file=True,val_auc_df='val_auc_df.csv',dev_auc_df='dev_auc_df.csv')
    if read_from_file:
        dev_auc_df = pd.read_csv(f'{results_path}{dev_auc_df}',index_col=0)
        val_auc_df = pd.read_csv(f'{results_path}{val_auc_df}',index_col=0)
        
        
    stats_list=['mean','std']#],'median','iqr','q1','min']
    title = '/'.join(results_path.split('/')[1:-2])
    num_seeds = max(dev_auc_df.shape[0],val_auc_df.shape[0])
    
    if num_seeds>1:
        val_auc_df.columns = val_auc_df.columns.map(string_split_handler)
        dev_auc_df.columns = dev_auc_df.columns.map(string_split_handler)
        
        plot_violin(val_auc_df,dev_auc_df,title=title,x_metric=x_metric,y_metric=y_metric,inner='quart',df_val_name=df_val_name,df_dev_name=df_dev_name)
        plot_violin(val_auc_df,dev_auc_df,title=title,x_metric=x_metric,y_metric=y_metric,inner='point',df_val_name=df_val_name,df_dev_name=df_dev_name)
        
        dev_auc_df=dev_auc_df.iloc[:,:-1]
        val_auc_df=val_auc_df.iloc[:,:-1]
        
        for col in dev_auc_df:
            dev_auc_df.loc[:,col] = dev_auc_df[col].apply(lambda x:None if x==0 else x)
        for col in val_auc_df:
            val_auc_df.loc[:,col] = val_auc_df[col].apply(lambda x:None if x==0 else x)
            
        dev_auc_stat = pd.DataFrame(columns = dev_auc_df.columns,index=stats_list)
        val_auc_stat = pd.DataFrame(columns = val_auc_df.columns,index=stats_list)

        for stat in stats_list:
            if stat == 'std':
                ylabel = f'STD ({y_metric})'
                dev_auc_stat.loc[stat,:] = dev_auc_df.std()
                val_auc_stat.loc[stat,:] = val_auc_df.std()
            elif stat=='q1':
                ylabel = f'Q1 ({y_metric})'
                dev_auc_stat.loc[stat,:] = dev_auc_df.quantile(q=0.25)
                val_auc_stat.loc[stat,:] = val_auc_df.quantile(q=0.25)
            elif stat=='min':
                ylabel = f'Minima ({y_metric})'
                dev_auc_stat.loc[stat,:] = dev_auc_df.min()
                val_auc_stat.loc[stat,:] = val_auc_df.min()
            elif stat=='iqr':
                ylabel = f'IQR ({y_metric})'
                dev_auc_stat.loc[stat,:] = dev_auc_df.quantile(q=0.75)-dev_auc_df.quantile(q=0.25)
                val_auc_stat.loc[stat,:] = val_auc_df.quantile(q=0.75)-val_auc_df.quantile(q=0.25)
            elif stat=='mean':
                ylabel = f'Average ({y_metric})'
                dev_auc_stat.loc[stat,:] = dev_auc_df.mean()
                val_auc_stat.loc[stat,:] = val_auc_df.mean()
            elif stat=='median':
                ylabel = f'Median ({y_metric})'
                dev_auc_stat.loc[stat,:] = dev_auc_df.median()
                val_auc_stat.loc[stat,:] = val_auc_df.median()
            
            stat_df = pd.concat([dev_auc_stat.loc[stat,:],val_auc_stat.loc[stat,:]],axis=1)
            stat_df.columns = [df_dev_name,df_val_name]
            stat_df=stat_df.round(4)
            #plot lines
            ax1 = stat_df.plot.line(ylabel=ylabel,xlabel=x_metric)
            ylim = plt.ylim()
            if stat_df.index.dtype=='float' or stat_df.index.dtype=='int':
                plt.title(f'{ylabel} vs. {x_metric}')
                plt.show()
            else:
                plt.close()
                #plot bars    
                ax2 = stat_df.plot.bar(ylabel=ylabel,xlabel=x_metric, ylim=ylim,rot='horizontal')
                plt.title(f'{ylabel} vs. {x_metric}')
                ax2.figure.subplots_adjust(bottom=0.17,top=0.95)
                ax2.yaxis.grid(True,which='minor')
                plt.show()
        stat_df_full = pd.concat([val_auc_stat,dev_auc_stat],axis=1,keys=[df_val_name,df_dev_name]).astype('float').round(4)
        return stat_df_full
#%%
    
if __name__ == '__main__':
    
    num_seeds = 1
    results_path = 'results/testing_ground/'
    
    sp_ensemble_size = 80
    number_of_KS_proteins = 50
    training_portion = 1
    train=False
    validate=False
    dropna=False

    patients_with_systemic_aes = ['GB-020-7000-NSCLC', 'GB-020-7002-NSCLC', 'GB-020-7006-NSCLC', 'GB-020-7008-NSCLC', 'GB-020-7009-NSCLC', 'GB-021-7102-NSCLC', 'GB-021-7103-NSCLC', 'GB-021-7105-NSCLC', 'GB-021-7110-NSCLC', 'GB-021-7112-NSCLC', 'GB-021-7113-NSCLC', 'GB-021-7117-NSCLC', 'GB-024-7400-NSCLC', 'GB-024-7404-NSCLC', 'GB-024-7415-NSCLC', 'GB-024-7418-NSCLC', 'GB-024-7419-NSCLC', 'GB-024-7421-NSCLC', 'GB-024-7422-NSCLC', 'GB-026-7601-NSCLC', 'GB-026-7602-NSCLC', 'GB-026-7603-NSCLC', 'GB-026-7604-NSCLC', 'GB-026-7606-NSCLC', 'GB-026-7607-NSCLC', 'GB-027-7705-NSCLC', 'GB-027-7708-NSCLC', 'GB-027-7713-NSCLC', 'GB-028-7805-NSCLC', 'GB-029-7902-NSCLC', 'GB-029-7908-NSCLC', 'GB-051-7194-NSCLC', 'GB-051-7198-NSCLC', 'IL-001-1002-NSCLC', 'IL-001-1022-NSCLC', 'IL-001-1028-NSCLC', 'IL-003-1300-NSCLC', 'IL-003-1304-NSCLC', 'IL-003-1318-NSCLC', 'IL-003-1320-NSCLC', 'IL-003-1330-NSCLC', 'IL-008-1709-NSCLC', 'IL-012-1610-NSCLC', 'IL-012-1646-NSCLC', 'IL-013-1800-NSCLC', 'IL-013-1804-NSCLC', 'IL-013-1810-NSCLC', 'IL-013-1816-NSCLC', 'IL-013-1828-NSCLC', 'IL-014-1910-NSCLC', 'IL-014-1913-NSCLC', 'IL-014-1914-NSCLC', 'IL-014-1915-NSCLC', 'IL-014-1924-NSCLC', 'US-031-6000-NSCLC', 'US-031-6002-NSCLC', 'US-031-6003-NSCLC', 'US-038-1010-NSCLC', 'US-038-1106-NSCLC', 'US-038-1154-NSCLC', 'US-152-1222-NSCLC']
    patients_with_gastro_aes = ['GB-020-7000-NSCLC', 'GB-020-7008-NSCLC', 'GB-020-7009-NSCLC', 'GB-021-7110-NSCLC', 'GB-021-7112-NSCLC', 'GB-021-7113-NSCLC', 'GB-021-7114-NSCLC', 'GB-024-7400-NSCLC', 'GB-024-7404-NSCLC', 'GB-024-7407-NSCLC', 'GB-024-7415-NSCLC', 'GB-024-7419-NSCLC', 'GB-024-7421-NSCLC', 'GB-024-7422-NSCLC', 'GB-026-7600-NSCLC', 'GB-026-7601-NSCLC', 'GB-026-7602-NSCLC', 'GB-026-7606-NSCLC', 'GB-026-7607-NSCLC', 'GB-027-7705-NSCLC', 'GB-027-7708-NSCLC', 'GB-027-7713-NSCLC', 'GB-028-7804-NSCLC', 'GB-029-7904-NSCLC', 'GB-029-7908-NSCLC', 'GB-051-7194-NSCLC', 'GB-051-7198-NSCLC', 'IL-001-1008-NSCLC', 'IL-001-1022-NSCLC', 'IL-003-1318-NSCLC', 'IL-003-1320-NSCLC', 'IL-008-1709-NSCLC', 'IL-012-1602-NSCLC', 'IL-012-1606-NSCLC', 'IL-013-1820-NSCLC', 'IL-014-1910-NSCLC', 'IL-014-1915-NSCLC', 'IL-014-1924-NSCLC', 'US-031-6000-NSCLC', 'US-038-1087-NSCLC', 'US-038-1141-NSCLC', 'US-152-1222-NSCLC']
    # read protein annotations and filters file
    protein_df = pd.read_csv(PROTEIN_FILTER_FILE, index_col=0, low_memory=False)
    protein_filt = list(protein_df.loc[protein_df['Sheba p-value'] > 0.05].index)
    
    df0 = read_data(DATA_PATH,index=0)
    if dropna:
        df0=df0.dropna(subset=[SETS,RESPONSE_METRIC])
    df = df0.loc[df0[SETS]=='Dev',:].iloc[:,:-1]
    
    #df = df.loc[[(idx in patients_with_systemic_aes) and not(idx in patients_with_gastro_aes) and df.loc[idx,RESPONSE_METRIC]==False for idx in df.index],:]
    
    val_df = df0.loc[df0[SETS]=='Val',:].iloc[:,:-1].rename({RESPONSE_METRIC:'Y'},axis=1)
    val_df['Y'] = val_df['Y'].apply(int)
        
    val_auc_df = pd.DataFrame(np.zeros((num_seeds,1)),columns = ['Val'], index=range(num_seeds))
    dev_auc_df = pd.DataFrame(np.zeros((num_seeds,1)),columns = ['Dev'], index=range(num_seeds))
    
    for seed in range(num_seeds):
        model_name = MODEL_NAME
        #model_name = f"{MODEL_NAME}_seed{seed}_filter{xcorr_filter_strength}"
        
        dev_df, validation_df, params = data_prepare_random_split(df=df, configuration_file=CONFIGURATION_FILE, seed=seed, train_ratio=training_portion)  
        validation_df = val_df
        if RESPONSE_METRIC=='ClinicalBenefit':
            dev_df['Y']=1-dev_df['Y']
            validation_df['Y']=1-validation_df['Y']
            
        params['sp_ensemble_size']=sp_ensemble_size
        params['number_of_KS_proteins']=number_of_KS_proteins
        # train the model
        if train:
            dev_auc_seed = model_generator(dev_df, RESPONSE_METRIC, model_name, seed, params)   
            dev_auc_df.loc[seed,'Dev'] = dev_auc_seed
        # predict the model
        if validate:
            #val_auc_seed, _, _, _, _, _, _ = model_predict(model_name, RESPONSE_METRIC, dev_df, flag_roc=1, flag_kaplan=0, flag_r2=1, flag_dev=1, flag_label_from_dev=0, title=RESPONSE_METRIC)    
            val_auc_seed, _, val_pred, _, _, _, _ = model_predict(model_name, RESPONSE_METRIC, validation_df, flag_roc=0, flag_kaplan=0, flag_r2=0, flag_dev=0, flag_label_from_dev=0, title=RESPONSE_METRIC)    
            val_auc_df.loc[seed,'Val'] = val_auc_seed
        
        
            
        
    #dev_auc_df.to_csv(f'{results_path}dev_auc_df.csv')
    #val_auc_df.to_csv(f'{results_path}val_auc_df.csv')
        
    #summary_plots(results_path,x_metric=values_title,read_from_file=True,val_auc_df='val_auc_df.csv',dev_auc_df='dev_auc_df.csv')
    
    
    print("All Done!!!")

        
        
 