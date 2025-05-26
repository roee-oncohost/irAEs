import json
import pickle
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import linregress
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from lifelines.statistics import logrank_test, proportional_hazard_test
from data_parser import prepare_v2_0, parse_x, parse_y, extract_xy_arrays

def youden_j_score(y_true, y_score):
    '''youden_j_score(y_true, y_score)
    Calclulate Youden's J statistic
    
    Parameters
    ----------
        y_true : list
            correct classes
        y_score : list
            prediction score
            
    Returns
    -------
    j_stat
        Youden's J statistic
    threshold
        The y_score threshold for Youden's J statistic
    fpr
        The false positive rate for Youden's J statistic
    '''
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_stats = tpr - fpr
    opt_point = np.argmax(j_stats)
    
    return j_stats[opt_point], thresholds[opt_point], fpr[opt_point]


def gmean_score(y_true, y_score):
    '''gmean_score(y_true, y_score)
    Calclulate the geometrical means of TPR, NPR at the optimal point on the
    ROC curve.
    
    Parameters
    ----------
        y_true : list
            correct classes
        y_score : list
            prediction score
            
    Returns
    -------
    j_stat
        Geometrical mean of TPR, NPR at the optimal point on the ROC curve.
    threshold
        The y_score threshold at optimal point
    fpr
        The false positive rate at optimal point
    '''
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gmeans_stat = np.sqrt(tpr * (1-fpr))
    opt_point = np.argmax(gmeans_stat)
    
    return gmeans_stat[opt_point], thresholds[opt_point], fpr[opt_point]


def pval_text(pval):
    '''pval_text(pval)
    Returns text for p-value in two formats 1) p < 0.01
                                            2) **
    
    Parameters
    ----------
        y_true : float
            p-value
    
    Returns
    -------
        tuple with two strings
    '''
    
    if pval >= 0.05:
        return '= {:.2f}'.format(pval), 'ns'
    elif pval >= 0.01:
        return '= {:.2f}'.format(pval), '*'
    elif pval >= 0.001:
        return '= {:.3f}'.format(pval), '**'
    elif pval >= 0.0001:
        return '< 0.001', '***'
    else:
        return '< 0.0001', '***'


def plot_roc_curve(y_true, y_score, thr=None, show_auc=True, show_pval=False, show_j_score=False, color=None, linewidth=2):
    '''plot_roc_curve(y_true, y_score, thr=None, show_pval=False, show_j_score=False)
    Display ROC curve for prediction data
    
    Parameters
    ----------
        y_true : list
            correct classes
        y_score : list
            prediction score
        thr : float
            threshold between classes. None for not displaying a specific threshold
        show_pval : bool
            if True calculates ROC curve p-value based on multiple repeats of random classifications for the given y values
        show_j_score : bool
            if True calculates and displays Youden's J statistic
                                              
    Example
    -------
        plot_roc_curve([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0.0, 0.1, 0.3, 0.7, 0.4, 0.5, 0.6, 0.2, 0.8, 0.9], thr=0.5, show_pval=False)
    '''
    
    if type(y_true) == list:
        y_true = np.array(y_true)
        
    if type(y_score) == list:
        y_score = np.array(y_score)
        
    auc = roc_auc_score(y_true, y_score)
    roc_crv = roc_curve(y_true, y_score)
    
    plt.plot(roc_crv[0], roc_crv[1], color=color, linewidth=linewidth)
    if thr is not None:
        y_pred = y_score >= thr
        sensitivity = recall_score(y_true, y_pred)
        specificity = recall_score(1-y_true, 1-y_pred)
        plt.plot([1-specificity], [sensitivity], 'og')
    else:
        sensitivity = None
        specificity = None
    plt.xlabel('1 - specificity'); plt.ylabel('sensitivity')
    plt.plot([0,1], [0, 1], 'r--', linewidth=2)
    plt.gca().set_aspect('equal', adjustable='box')
    
    txt = ''
    if show_auc:
        txt += 'AUC = {:.2f}'.format(auc)

    REPEATS = 20000
    auc_rand = np.zeros(REPEATS)
    for i in range(REPEATS):
        auc_rand[i] = roc_auc_score(y_true, np.random.uniform(size=len(y_score)))
    pval = sum(auc_rand > auc) / len(auc_rand)
    if show_pval:
        txt += '\np-value ' + pval_text(pval)[0]
    if show_j_score:
        youden_j, opt_thr, fpr = youden_j_score(y_true, y_score)
        txt = txt + '\nYouden''s J = {:.1f}\nOptimal thr = {:.3}'.format(youden_j, opt_thr)
        
        plt.plot([fpr, fpr], [fpr, fpr+youden_j])
        
    plt.text(0.70, 0.20, txt, ha='center', size=20)
    return pval


def plot_kaplan_meier(data: DataFrame, duration: str, event: str, group: str=None, order=None, show_legend=True, show_n=True, show_censors=True, show_ci=True, linewidth=2, alpha=1., color=['#0AA499', '#EF4557', '#453D3C'], month_length=30.437):
    ''' Plots Kaplan-Meier comparing groups of patients
    
    Parameters
    ----------
        data : DataFrame
            a dataframe containing survival data for a set of patients
        duration : str
            numeric column containing duration of survival per subject (in days)
        event : str
            a boolean column indicating event
                0 indicates censorship event (subject withdrawn),
                1 indicates death\\progression event
        group : str
            a categorical column indicating division of dataset into groups
                                              
    Example
    -------
    duration = [10, 20, 30, 50, 80, 120, 60, 100, 150, 200, 250, 300, 300, 300]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
    group = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    survival_df = pd.DataFrame(zip(duration, event, group), columns=['Duration', 'Event', 'Group'])
    survival_df['GroupName'] = survival_df['Group'].map({1: 'PROphet Positive', 0: 'PROphet Negative'})
    survival_df['Disease'] = 'NSCLC'
    
    plt.figure(figsize=(8,6))
    plot_kaplan_meier(data=survival_df, duration='Duration', event='Event', group='Disease', color='g', show_censors=True, show_ci=True, linewidth=2)
    
    plt.figure(figsize=(8,6))
    plot_kaplan_meier(data=survival_df, duration='Duration', event='Event', group='GroupName', order=['PROphet Positive', 'PROphet Negative'], color={'PROphet Positive': 'b', 'PROphet Negative': 'r'}, show_censors=True, show_ci=True, linewidth=2)
    
    plt.figure(figsize=(8,6))
    plot_kaplan_meier(data=survival_df, duration='Duration', event='Event', group='GroupName', order=['PROphet Positive', 'PROphet Negative'], color={'PROphet Positive': 'b', 'PROphet Negative': 'r'}, show_censors=False, show_ci=False, linewidth=3)
    '''
    
    if type(data) != pd.core.frame.DataFrame:
        raise ValueError('data must be a dataframe')
    cols = [duration, event]
    
    if group is not None:
        cols = cols + [group]
    for col in cols: 
        if type(col) != str:
            raise ValueError(f'{col} must be a column name')
        if col not in data.columns:
            raise ValueError(f'{col} column missing from data')
    df = deepcopy(data[cols])
    if group is None:
        group = '_group_'
        df[group] = ''
    unique_groups = df[group].unique()
    if order is None:
        order = unique_groups
    else:
        if len(order) < len(unique_groups):
            raise ValueError(f'All values in {group} column must be in order')
        for grp in unique_groups:
            if grp not in order:
                raise ValueError(f'{grp} appears {group} but missing from order')

    if show_n:
        grp_display_names = {grp: str(grp) + ' (n={})'.format(sum(df[group]==grp)) for grp in unique_groups}
    else:
        grp_display_names = {grp: str(grp) for grp in unique_groups}
        
    if type(linewidth) == int or type(linewidth) == float:
        linewidth = {grp: linewidth for grp in order}
    if type(linewidth) != dict:
        raise ValueError('linewidth must be a numeric scalar, or a dict with a value for each group')
    if len(color) < len(order):
        raise ValueError(f'color must contain a value for each group (len(linewidth) = {len(linewidth)} < {unique_groups} groups in data)')
    if type(color) == str:
        color = {grp: color for i, grp in enumerate(order)}
    if type(color) == list:
        color = {grp: color[i] for i, grp in enumerate(order)}
    
    # Plot survival curve for each group
    kmf = []
    for i, grp in enumerate(reversed(order)):
        kmf.append(KaplanMeierFitter())
        if sum(df[group]==grp) > 0:
            # Fit Kaplan Meier to data separately for each group
            kmf[i].fit(durations=df.loc[df[group]==grp, duration]/month_length, event_observed=df.loc[df[group]==grp, event])
            
            # Plot Kaplan Meier separately for each group
            kmf[i].plot_survival_function(show_censors=show_censors, censor_styles={'marker': '|', 'mew': 1.5, 'ms': 7}, ci_show=show_ci, linewidth=linewidth[grp], alpha=alpha, color=color[grp])
    plt.xlabel('months passed'); plt.ylabel('survival probability')
    
    # Manage legend
    if show_legend:
        # Handle cases where no data was sensored
        warnings.filterwarnings('ignore')
        lgnd = []
        for grp in reversed(order):
            if sum(df[group]==grp) > 0:
                hide_ci = int(show_ci)
                censors_exist = not(all(df.loc[df[group]==grp, event] == 1))
                hide_censors = int(show_censors) * int(censors_exist)
                lgnd += ['_nolegend_'] * hide_censors + [grp_display_names[grp]] + ['_nolegend_'] * hide_ci
        #print(lgnd)
        plt.legend(lgnd)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), [grp_display_names[grp] for grp in order])
        warnings.filterwarnings('default');
    else:
        ax = plt.gca()
        ax.get_legend().remove()
            
            
def cox_model_summary(data: DataFrame, duration: str, event: str, group: str, order=None, drop_missing_values=False, verbose=True):
    '''cox_model_summary(groups, duration, event, verbose=True)
    Calculate summary statistics comparing survival of two groups
    
    Parameters
    ----------
        data : DataFrame
            a dataframe containing survival data for a set of patients
        duration : str
            numeric column containing duration of survival per subject
        event : str
            a boolean column indicating event
                0 indicates censorship event (subject withdrawn),
                1 indicates death\\progression event
        group : int
            a categorical column indicating division of dataset into groups
                                              
    Returns
    -------
    dict
        A dictionary containing cox model results and log rank test results.
        Cox model results include harzard ratio, hazard ratio confidence
        intervals, and multiple test p-values for the cox model:
            - log likelihood ratio test: the p-value for the cox model
            - wald_test: the p-value for a specific variable in the model
                    (in the case of two groups only a single p-value)
            - proportional_hazard_test: p-value for rejection of the
                    proportional hazard assumption (in case this assumption is
                    rejected, using the cox model may be problematic)
            
    Example
    -------
    duration = [10, 20, 30, 50, 80, 120, 60, 100, 150, 200, 250, 300, 300, 300]
    event = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
    group = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    survival_df = pd.DataFrame(zip(duration, event, group), columns=['Duration', 'Event', 'Group'])
    cox_model_summary(data=survival_df, duration='Duration', event='Event', group='Group')
    '''
    
    if type(data) != pd.core.frame.DataFrame:
        raise ValueError('data must be a dataframe')
    cols = [duration, event, group]
    
    for col in cols: 
        if type(col) != str:
            raise ValueError(f'{col} must be a column name')
        if col not in data.columns:
            raise ValueError(f'{col} column missing from data')
    # Coren bug fix
    if data[event].dtype != np.dtype('float64'):
        raise ValueError(f'Event column ({event}) must be of type float.')
    if data[duration].dtype == np.dtype('int64'):
        data[duration]=data[duration].astype('float')
    if data[duration].dtype != np.dtype('float64'):
        raise ValueError(f'Duration column ({duration}) must be of type float.')
    #
        
    cox_data = deepcopy(data[cols])

    # check for order of groups
    unique_groups = data[group].unique()
    if order is None:
        order = unique_groups
    else:
        if len(order) < len(unique_groups):
            raise ValueError(f'All values in {group} column must be in order')
        for grp in unique_groups:
            if grp not in order:
                raise ValueError(f'{grp} appears {group} but missing from order')

    # convert group categorical column to numeric
    groups = order
    cox_data[group] = cox_data[group].map({grp: grp_idx for grp_idx, grp in enumerate(groups)})
    groups = cox_data[group].unique()
    
    if drop_missing_values:
        cox_data = cox_data.dropna()
    cph = CoxPHFitter()
    cph.fit(df=cox_data, duration_col=duration, event_col=event)
    log_likelihood_test = cph.log_likelihood_ratio_test()
    wald_test_pval = cph._compute_p_values()[0]
    pht = proportional_hazard_test(cph, cox_data, time_transform='rank')
    if pht.p_value[0] > 0.05:
        assumptions = 'Proportional hazard assumption: OK'
    else:
        assumptions = 'Proportional hazard assumption: failed ({})'.format(pht.p_value[0])
    
    # Run univariate cox regression
    hr = cph.hazard_ratios_[group]
    hr_ci = cph.confidence_intervals_.iloc[0]
    
    # If the contrast is between two groups
    if len(groups) != 2:
        logrank_result_pval = np.nan
    else:
        # Run log-rank test for difference in survival distribution
        logrank_result = logrank_test(cox_data.loc[cox_data[group]==groups[0], duration], cox_data.loc[cox_data[group]==groups[1], duration],
                                      cox_data.loc[cox_data[group]==groups[0], event], cox_data.loc[cox_data[group]==groups[1], event])
        logrank_result_pval = logrank_result.p_value
    
    if verbose:
        print(('HR = {:.2f} (95% CI: {:.2f}-{:.2f})\n' \
              + 'Logrank test p {}\nLikelihood ratio p {}\n' \
              + 'Wald test p {} ({})').format(hr, np.exp(hr_ci[0]), np.exp(hr_ci[1]),
                                                pval_text(logrank_result.p_value)[0],
                                                pval_text(log_likelihood_test.p_value)[0],
                                                pval_text(wald_test_pval)[0], assumptions))
    
    p_values = {'logrank_test': logrank_result_pval, 'log_likelihood_test': log_likelihood_test.p_value, 'wald_test': wald_test_pval, 'proportional_hazard_test': pht.p_value[0]}
    return {'hr': hr, 'hr_ci': [np.exp(hr_ci[0]), np.exp(hr_ci[1])], 'p_values': p_values}


def generate_rap_table(rap_model, evaluation_df, thr):
    '''generate_rap_table(rap_model, evaluation_df)
    Generate a table with features as columns and evaluation subjects as
    rows. For each feature and subject the number of iterations where the
    feature passed the KS test and passed the RAP threshold for the patient.
    

    Parameters
    ----------
    rap_model : RapResponsePredictor
        A RapResponsePredictor model
    evaluation_df : DataFrame
        A dataframe containing the feature measurements for all patients in the
        evaluation set (dev set or validation set). SubjectId column contains
        the patient identifiers (e.g., IL-001-1001-NSCLC).
    horizon : str
        Horizon to evaluate (T0 or T1).

    Returns
    -------
    DataFrame
        a table with features as columns and evaluation subjects as
        rows. For each feature and subject the number of iterations where the
        feature passed the KS test and passed the RAP threshold for the patient.
    '''
    
    features = rap_model.proteins
    patients = evaluation_df.index

    # Generate empty rap table
    rap_table = pd.DataFrame(np.zeros((len(patients), len(features))))
    rap_table.index = patients
    rap_table.columns = features

    for iter_idx, iter_models in enumerate(rap_model.sp_models):
        for feature_idx, feature in enumerate(iter_models['xgb_models'].keys()):
            model = iter_models['xgb_models'][feature]
            #test_dict = extract_xy_arrays(rap_model.sp_features + [feature], evaluation_df)
            test_dict = extract_xy_arrays([feature] + rap_model.sp_features, evaluation_df)
            dtest = xgb.DMatrix(test_dict['X'])
            pred = model.predict(dtest)
            rap_table.loc[:, feature] += pred > thr
            
    # Return sorted rap table
    return rap_table.iloc[:, np.argsort(np.sum(rap_table, axis=0))[::-1]]
    

def pred_vs_obs_prob(actual, pred, index=None, interval=0.1, plot=True):
    # Check valid input
    interval_error_message = 'Interval must be float (probability; between 0 and 1) or int (n nearest patients; 2 or greater)'
    if type(interval) == float:
        if (interval <= 0.) | (interval >= 1.):
            raise ValueError(interval_error_message)
    elif type(interval) == int:
        if interval <= 1:
            raise ValueError(interval_error_message)
    else:
        raise ValueError(interval_error_message)
             
    df = pd.DataFrame(zip(actual, pred), columns=['actual', 'pred'], index=index)
    
    df['obs'] = np.nan
    for patient in df.index:
        patient_pred = df.loc[patient, 'pred']
        if type(interval) == float:
            adjacent_patients = (df['pred'] > patient_pred - interval) & (df['pred'] < patient_pred + interval)
            df.loc[patient, 'obs'] = df.loc[adjacent_patients, 'actual'].mean()
        elif type(interval) == int:
            adjacent_patients = (df.loc[df.index != patient, 'pred'] - patient_pred).abs().sort_values()[:interval].index
            df.loc[patient, 'obs'] = df.loc[adjacent_patients, 'actual'].mean()
    
    return df
    

def pred_vs_obs(pred, actual, interval=.1, r_square_fit=False, show_pval=True, aparametric_pval=False,
                     aparametric_pval_iter=20001,
                     response_metric_legend={'ORR1_early_death': '3 months', 'ORR2_early_death': '6 months','DCB at 1 year' : '1 year'}, 
                     plot=False):
    '''
    Plot predicted vs observed response probabilites over multiple response
    metrics.
    
    Parameters
    ----------
    pred : dict of lists
        Dictionary with response metrics as keys and lists of predicted
        response probabilities as values.
    actual : dict of lists
        Dictionary with response metrics as keys and lists of actual
        responses as values.
    interval : float
        Value between 0 and 1 representing average range for calculating
        observed probability based on actual response.
    r_square_fit : boolean, optional
        If True, calculate R square after linear fit. Calculate R square
        relative to pred=obs otherwise. The default is True.
    aparametric_pval  boolean, optional
        If True, calculate aparametric p-value for R square based on
        kernel density estimation (slow performance). Calculate parameeric
        p-value if False. The default is False.

    Returns
    -------
    obs
        Observed predicted values
        
    r_sq
        R squared of actual distribution
    
    shuffled_r_sq
        R squared of shuffled distributions (or None if aparametric_pva=False)

    '''
    # Perform linear regression for predicted vs observed response probability over all response metrics
    response_metrics = pred.keys()
    pred_obs = {m: [] for m in response_metrics}
    obs = {m: [] for m in response_metrics}
    for rm in response_metrics:
        pred_obs[rm] = pred_vs_obs_prob(actual[rm], pred[rm], interval=interval)
        obs[rm] = pred_obs[rm]['obs']
        
    all_preds = np.concatenate([pred[m] for m in response_metrics])
    all_obs = np.concatenate([obs[m].values for m in response_metrics])
    slope, intercept, _, p_value, std_err = linregress(all_preds, all_obs)

    r_sq = r_squared(all_preds, all_obs, fit=r_square_fit)

    # Calculare aparametric p-value for R-square
    shuffled_r_sq = np.zeros(aparametric_pval_iter)
    
    if show_pval and aparametric_pval:
        print('Calculating aparametric p-value for R square')
        for i in range(aparametric_pval_iter):
            actual_resampled = {m: [] for m in response_metrics}
            if i > 0 and (i+1) % 10 == 0:
                print(f'Iteration {i+1} of {aparametric_pval_iter}')
            pred_obs_resampled = {}
            for rm in response_metrics:
                actual_resampled[rm] = np.random.permutation(actual[rm])
                pred_obs_resampled[rm] = pred_vs_obs_prob(actual_resampled[rm], pred[rm], interval=interval)
            all_preds_sampled = np.concatenate([pred_obs_resampled[m]['pred'].values for m in response_metrics])
            all_obs_sampled = np.concatenate([pred_obs_resampled[m]['obs'].values for m in response_metrics])
            shuffled_r_sq[i] = r_squared(all_preds_sampled, all_obs_sampled, fit=r_square_fit)
            
        pd.Series(shuffled_r_sq).to_csv('r_values_for_shuffled_data.csv')
        p_value = (sum(shuffled_r_sq > r_sq)+1) / aparametric_pval_iter
        p_value = 2*p_value # Two tailed test
        print(f'aparametric p-value for R square = {p_value}')
        
        if plot:
            plt.figure(figsize=(8,6))
            plt.hist(shuffled_r_sq)
            ax = plt.gca()
            yl = ax.get_ylim()
            plt.plot([r_sq, r_sq], yl, 'r--')
            plt.legend(['Actual data', 'Shuffled labels'])
    else:
        shuffled_r_sq = None
    
    if plot:
        # Display predicted vs observed response probability
        plt.figure(figsize=(8,6))
        range_ = np.array([np.min(all_preds), np.max(all_preds)])
        plt.plot([0, 1], [0, 1], 'k', linewidth=1)
        for rm in response_metrics:
            if len(pred_obs[rm]) > 0:
                plt.plot(pred_obs[rm]['pred'], pred_obs[rm]['obs'], 'o', markersize=10)
        if r_square_fit:
            plt.plot(range_, slope*range_+intercept, 'k--', linewidth=2)
        plt.xlabel('Predicted AE probability')
        plt.ylabel('Observed AE probability')
        plt.title(list(response_metric_legend.values())[0])
        #plt.legend(np.append(['_nolegend_'], pd.Series(response_metrics).map(response_metric_legend).values),
        #           loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.xlim([-0.04, 1.04]); plt.ylim([-0.04, 1.04])
        if show_pval:
            plt.text(0.98, 0.02, f'$R^{2}$ = {r_sq:.2f}\np-value {pval_text(p_value)[0]}', va='bottom', ha='right')
        else:
            plt.text(0.98, 0.02, f'$R^{2}$ = {r_sq:.2f}', va='bottom', ha='right')
        plt.show()
    return obs, r_sq, shuffled_r_sq
    
    
def r_squared(x, y, fit=True):
    if fit:
        slope, intercept, _, _, _ = linregress(x, y)
        ypred = slope*x+intercept
    else:
        ypred = x
    res = y-ypred
    SSres = np.sum(np.power(res, 2))
    SStot = np.sum(np.power(y-np.mean(y), 2))
    return 1 - (SSres/SStot)


if __name__ == '__main__':    
    # Parameters
    DATASET_VERSION = '1.1' # Supported: 1.1, 2.0
    if DATASET_VERSION == '1.1':
        DATA_FILE = './data/Classifier_v1.1g_FILTERED_full.csv'
        VERSION_TEXT = 'v1.1g'
    elif DATASET_VERSION == '2.0':
        DATA_FILE = './data/230101 ClinMeas_v2.0_DevChemo.csv'
        VERSION_TEXT = 'v2.0a'
    MODEL_FILE_FORMAT = './models/models_file_{version}_FILTERED_full_{rm}_horizon_{horizon}_seed_{seed}.pkl'
    CONFIGURATION_FILE = 'response_prediction_configuration.json'
    EXCLUDE_ADVANCED_LINE_PATIENTS_FROM_VALIDATION_SET = False
    Y_PRED_COL = 'y_pred_sp_scaled' # 'y_pred_sp', 'y_pred_sp_scaled', 'y_pred_cm'
    EVALUATE = 'test' # Options: 'test', 'validation', 'test&validation'
    SEED=3
    HORIZON = 'T0' # ['T0', 'T1']
    RESPONSE_METRICS = ['ORR1_early_death', 'ORR2_early_death', 'DCB at 1 year']
    RESPONSE_METRIC_DICT = {'ORR1_early_death': '3 months', 'ORR2_early_death': '6 months', 'DCB at 1 year': '1 year'}
    R_SQUARED_INTERVAL = .15
    R_SQUARED_PVAL_N_ITERATIONS = 10
    PLOT_KAPLAN_MEIER = True
    
    survival_df = {m: [] for m in RESPONSE_METRICS}
    raps_counts = {m: [] for m in RESPONSE_METRICS}
    for rm in RESPONSE_METRICS:
        print(f'Response metric: {rm}')
        # read configuration file
        with open(CONFIGURATION_FILE,'r') as f:
            conf = json.load(f)['rap_response_predictor']
        # prepare Y column
        globl = conf['globals']
        y_map = {'R': globl['RESPONDER'], 'SD': globl['STABLE'], 'NR': globl['NONRESPONDER']}
        # read datafile
        xy_df = pd.read_csv(DATA_FILE, low_memory=False, index_col='SubjectId')
        # prepare clinical data columns
        if DATASET_VERSION == '1.1':
            xy_df.drop(labels='Unnamed: 0', axis=1, inplace=True)
        elif DATASET_VERSION == '2.0':
            xy_df = prepare_v2_0(xy_df)
        xy_df = parse_x(xy_df)
        dev_df, validation_df = parse_y(xy_df, rm, y_map=y_map, drop_unknown_y=True)

        if EVALUATE == 'test':
            evaluation_df = dev_df
        elif EVALUATE == 'validation':
            evaluation_df = validation_df
        elif EVALUATE == 'test&validation':
            evaluation_df = pd.concat([dev_df, validation_df])
        
        # Load model
        model_file = MODEL_FILE_FORMAT.format(version=VERSION_TEXT, rm=rm, horizon=HORIZON, seed=SEED)
        
        # Exclude advanced line patients from model evaluation
        print(f'Validation set size: {len(evaluation_df)}')
        if EXCLUDE_ADVANCED_LINE_PATIENTS_FROM_VALIDATION_SET:
            evaluation_df = evaluation_df.loc[evaluation_df['Line'] == 0]
            print(f'Validation set size after excluding advanced line patients: {len(evaluation_df)}')
        
        # Load model file
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        threshold = model.sp_p_threshold
            
        # Extract original test results from model file
        if type(model.prediction) == dict:
            test_pred = pd.DataFrame.from_dict(model.prediction, orient='index')
        else:
            test_pred = deepcopy(model.prediction)
            
        if EVALUATE != 'test':
            validation_pred = model.predict(validation_df)
        
        # Generate survival df for validation set
        SURVIVAL = 'OS' # 'OS' or 'PFS'
        if SURVIVAL == 'OS':
            DURATION = 'OS Duration'        # 'PFS right censored' or 'OS duration'
            EVENT = 'OS Event'              # 'PFS event' or 'OS event'
            SURVIVAL_TITLE = 'OS rate'
        elif SURVIVAL == 'PFS':
            DURATION = 'PFS Duration' # 'PFS right censored' or 'OS duration'
            EVENT = 'PFS Event'             # 'PFS event' or 'OS event'
            SURVIVAL_TITLE = 'PFS rate'
        
        survival_df[rm] = evaluation_df[[DURATION, EVENT, rm, 'Sex', 'PDL1_1_49', 'PDL1_more_50', 'Line']].copy()
        #survival_df.set_index('SubjectId', inplace=True)
        survival_df[rm].columns = [DURATION, EVENT, 'y', 'Sex', 'PDL1_1_49', 'PDL1_more_50', 'Line']
        if type(survival_df[rm][DURATION][0]) == str:
            survival_df[rm].loc[~survival_df[rm][DURATION].str.isnumeric(), DURATION] = np.nan
            survival_df[rm][DURATION] = survival_df[rm][DURATION].astype(float)
        if type(survival_df[rm][EVENT][0]) == str:
            survival_df[rm].loc[~survival_df[rm][EVENT].str.isnumeric(), EVENT] = np.nan
        survival_df[rm]['y'] = survival_df[rm]['y'].map(y_map)
        survival_df[rm] = survival_df[rm].loc[(~survival_df[rm]['y'].isna()) & (~survival_df[rm][DURATION].isna()) & (~survival_df[rm][EVENT].isna())]    
        survival_df[rm][EVENT] = survival_df[rm][EVENT].astype(int)
        survival_df[rm]['Center'] = survival_df[rm].index.str.slice(3, 6)
        survival_df[rm]['PDL1'] = survival_df[rm]['PDL1_1_49'] + 2*survival_df[rm]['PDL1_more_50']
        
        pred_cols = model.prediction.columns[model.prediction.columns.str.startswith('y_pred')] # Extracts following columns: 'y_pred_sp', 'y_pred_sp_scaled', 'y_pred_cm'
        if EVALUATE == 'test':
            survival_df[rm] = survival_df[rm].join(test_pred[pred_cols])
        elif EVALUATE == 'validation':
            survival_df[rm] = survival_df[rm].join(validation_pred[pred_cols])
        elif EVALUATE == 'test&validation':
            survival_df[rm] = survival_df[rm].join(
                pd.concat([test_pred[pred_cols],
                            validation_pred[pred_cols]]))
        
        # Plot ROC curve
        plt.figure(figsize=(8,6));
        plt.rcParams.update({'font.size': 20})
        plot_roc_curve(survival_df[rm]['y'], survival_df[rm][Y_PRED_COL], thr=None, show_pval=True, show_j_score=False)
        plt.title(RESPONSE_METRIC_DICT[rm])
        plt.savefig(fr'figures\ROC_curve_{rm}.pdf')
        plt.savefig(fr'figures\ROC_curve_{rm}.png')
        plt.show()
        
        if PLOT_KAPLAN_MEIER:
            # Print summary statistics
            survival_df[rm]['group_id'] = (survival_df[rm][Y_PRED_COL] >= threshold).astype(int)
            survival_df[rm]['group'] = (survival_df[rm][Y_PRED_COL] >= threshold).map({True: 'Prolonged benefit',
                                                                                       False: 'Limited benefit'})
            hr_summary = cox_model_summary(data=survival_df[rm], duration=DURATION, event=EVENT,
                                            group='group_id')
            
            # Plot Kaplan-Meier survival curve
            color = ['#0AA499', '#EF4557']
            linewidth = 3
                
            plt.figure(figsize=(8,6));
            plt.rcParams.update({'font.size': 20})
            plot_kaplan_meier(data=survival_df[rm], duration=DURATION, event=EVENT,
                                            group='group', order=['Prolonged benefit', 'Limited benefit'],
                                            show_ci=False, show_censors=False, linewidth=linewidth, color=color)
            plt.title('HR = {:.2f} (95% CI: {:.2f}-{:.2f}), p {}'.format(hr_summary['hr'], hr_summary['hr_ci'][0], hr_summary['hr_ci'][1], pval_text(hr_summary['p_values']['logrank_test'])[0]))
            plt.ylabel(SURVIVAL_TITLE)
            ax=plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(fr'figures\Kaplan-Meier_{rm}_{SURVIVAL}.pdf')
            plt.savefig(fr'figures\Kaplan-Meier_{rm}_{SURVIVAL}.png')
            plt.show()
        
        # Extract model RAP counts (as decided during model generation)
        raps_counts[rm] = pd.DataFrame.from_dict(model.RAPs, orient='index').sort_values(by=0, ascending=False)
        
        # Generate RAP table (based on actual RAPs per patient in final dataset)
        rap_table = generate_rap_table(model, evaluation_df, threshold)
        rap_max_repeats = pd.DataFrame(rap_table.max(axis=0)).transpose()
        rap_max_repeats.index = ['Repeats']
        rap_table = pd.concat([rap_max_repeats, rap_table])
        rap_table = rap_table.sort_values(by='Repeats', axis=1, ascending=False)
        rap_table.to_csv(fr'output\rap_table_{rm}.csv')
        
        print('') # New line for next response metric
        
    writer = pd.ExcelWriter('./output/RAP counts.xlsx', engine='xlsxwriter')
    for sheet in raps_counts.keys():
        raps_counts[sheet].to_excel(writer, sheet_name=sheet)
    writer.save()
    
    # Calculate predicted vs observed response pobability
    pred_obs = {m: [] for m in RESPONSE_METRICS}
    pred = {m: [] for m in RESPONSE_METRICS}
    actual = {m: [] for m in RESPONSE_METRICS}
    obs = {m: [] for m in RESPONSE_METRICS}
    for rm in RESPONSE_METRICS:
        pred[rm] = survival_df[rm][Y_PRED_COL]
        actual[rm] = survival_df[rm]['y']
    
    # Display predicted vs observed response
    plt.figure()
    obs, rsq, rsq_shuffled = pred_vs_obs(pred, actual, interval=R_SQUARED_INTERVAL,
                                         r_square_fit=False, show_pval=False, aparametric_pval=True, aparametric_pval_iter=20001,
                                         plot=True, response_metric_legend=RESPONSE_METRIC_DICT) # For showing p < 0.0001 use aparametric_pval_iter=20001
    plt.savefig(fr'figures\Pred-obs_interval{R_SQUARED_INTERVAL:.2f}.pdf')
    plt.savefig(fr'figures\Pred-obs_interval{R_SQUARED_INTERVAL:.2f}.png')
    rsq_df = pd.DataFrame(rsq_shuffled, columns=['shuffled'])
    rsq_df['actual']=np.nan; rsq_df.loc[0, 'actual']=rsq
    rsq_df.to_csv(fr'output\rsquare_{R_SQUARED_INTERVAL:.2f}.csv')
    
    # Generate predictions
    predictions = None
    for rm in RESPONSE_METRICS:
        obs[rm].index = actual[rm].index
        rm_predictions = pd.DataFrame(actual[rm]).join(pred[rm])
        rm_predictions = rm_predictions.join(obs[rm])
        rm_predictions.columns = [rm + '_actual', rm + '_pred', rm + '_obs']
        plt.figure()
        sns.swarmplot(x=rm_predictions.iloc[:,0].map({0.: 'NR', 1.: 'R'}), y=rm_predictions.iloc[:,1], order=['R', 'NR'])
        plt.ylim([0,1])
        if predictions is None:
            predictions = rm_predictions
        else:
            predictions = predictions.join(rm_predictions)
    predictions.to_csv(r'output\predictions.csv')
    
    # Plot correlations between early and late timepoints
    plt.figure(figsize=(8,6));
    plt.rcParams.update({'font.size': 20})
    rm1 = 'ORR1_early_death' # Early timepoint: 'ORR1_early_death', 'ORR2_early_death'
    rm2 = 'ORR2_early_death' # Late timepoint: 'ORR2_early_death', 'DCB at 1 year'
    predictions['Actual Response'] = (predictions[rm1 + '_actual'] + 2*predictions[rm2 + '_actual']).map({0: 'NR both', 1: 'R early, NR late', 2: 'NR early, R late', 3: 'R both'})
    include = ~predictions['Actual Response'].isna()
    plt.scatter(predictions.loc[include, rm1 + '_pred'], predictions.loc[include, rm2 + '_pred'], c=predictions.loc[include, 'Actual Response'].map({'NR both': 'r', 'R both': 'g', 'R early, NR late': 'b', 'NR early, R late': 'k'}).values)
    plt.plot([0,1], [0, 1], 'k--', linewidth=2)
    plt.xlabel(RESPONSE_METRIC_DICT[rm1] + ' prediction')
    plt.ylabel(RESPONSE_METRIC_DICT[rm2] + ' prediction')
    plt.savefig(fr'figures\Pred-early-late-scatter-{rm1}-{rm2}.pdf')
    plt.savefig(fr'figures\Pred-early-late-scatter-{rm1}-{rm2}.png')