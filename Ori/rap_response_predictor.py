##### IMPORTS - EXTERNAL #####
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
# from scipy.optimize import minimize # Add in case would like to add Huber regression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RANSACRegressor#, HuberRegressor
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.multitest import multipletests
from copy import deepcopy
from logger_manager import getLogger
import lifelines

from data_parser import extract_xy_arrays


##### DEFAULT INIT VALUES #####
PREDICTOR_VERSION = 'v_test_dec_2023'

XGB_DEFAULT_PARAMS = {'max_depth': 3, 'eta': 0.8, 'lambda': 5, 'alpha': 2, 'num_parallel_tree': 3, 
              'objective': 'binary:logistic', 'eval_metric': 'logloss'}

CLASS_INITIALIZATION_VALUES = {
    'sp_xgb_params': XGB_DEFAULT_PARAMS, 'sp_xgb_num_round': 10, 'sp_train_ratio': 0.75, 'sp_ensemble_size': 80, 
    'cm_xgb_params': XGB_DEFAULT_PARAMS, 'cm_xgb_num_round': 10, 'cm_train_ratio': 0.75, 'cm_ensemble_size': 80, 
    'proteins': [], 'sp_features': [], 'clinical_features': [],
    'number_of_KS_proteins': 0, 'sp_p_threshold': 0,
    'sp_scaling_method': 'largest_span', 'sp_scaling_points': 10,
    'n_expected_outliers': 2,
    'n_first_ks': 1,
    'multipletests_method': 'fdr_bh', 'multipletests_alpha': 0.5}

logger = getLogger('RapResponsePredictor')

##### CLASS #####
class RapResponsePredictor:
    """
    Response predictor for cancer patients, based on the identification of Resistance Asociated Proteins
    Main methods:
        fit(train_df): generate prediction model
        predict(test_df): evaluate response probability
    """
    def __init__(self, **kwargs):
        print('blablabla333333') # TODO: Remove, just to make sure the right version is running
        # set default values    
        for item, value in CLASS_INITIALIZATION_VALUES.items():
            self.__dict__[item] = deepcopy(value)
        # initialize configuration arguments
        for item, value in kwargs.items():
            self.__dict__[item] = deepcopy(value)
        # verify values
        self.eval_initialized_values()
        logger.info(f'initiated rapResponsePredictor module {PREDICTOR_VERSION}')
        parameter_values = 'initialized parameteres:'
        for key, value in self.__dict__.items():
            if key == 'proteins':
                logger.info(f'Total of {len(value)} proteins\nFirst proteins: {value[:5]})')
                logger.debug(f'Full protein list: {value}')
            else:
                parameter_values = parameter_values + f'\n{key}: {value}'
        logger.info(parameter_values)
        # initialize fit property values
        self.prediction = {}
        self.train_ids = []
        self.y_train = []
        self.sp_models = []
        self.clinical_models = []
        self.RAPs = {}
        self.sp_auc = np.nan
        self.sp_mean_auc = np.nan
        self.sp_median_auc = np.nan
        self.cm_auc = np.nan
        self.iteration_predictions = []
        self.version = PREDICTOR_VERSION
        # flag that fit was not yet performed on this class instance
        self.is_fitted = False
        
        
    ##### FIT #####    
    def fit(self, dev_df):
        """
        generate response prediction model
        input parameters : 
            train_df (pandas dataframe) - including proteomic features, clinical features, actual response
        """
        logger.info('initiating model fitting')
        self.eval_dev_df(dev_df)
        self.train_ids = list(dev_df.index)
        
        self.y_train = np.array(dev_df['Y'])
        if self.sp_p_threshold == "mean_response_rate":
            self.sp_p_threshold = np.mean(self.y_train)
        self.prediction = {patient: {'y': y, 'y_pred_sp': [], 'y_pred_sp_mean': [], 'y_pred_sp_median': [], 'y_pred_sp_scaled': np.nan, 'y_pred_sp_mean_scaled': np.nan, 'y_pred_cm': []} \
                           for patient, y in zip(self.train_ids, self.y_train)}
        self.prediction = pd.DataFrame.from_dict(self.prediction, orient='index')
        for iteration in range(self.sp_ensemble_size):
            self.generate_sp_model(dev_df, iteration)
        self.average_prediction('y_pred_sp')
        self.average_prediction('y_pred_sp_mean')
        self.average_prediction('y_pred_sp_median')
        if len(self.clinical_features) > 0:
            self.generate_clinical_model(dev_df)
            self.average_prediction('y_pred_cm')
        self.extract_sp_scaling_factors(metric='y_pred_sp')
        self.extract_sp_scaling_factors(metric='y_pred_sp_mean')
        self.evaluate_auc()
        self.is_fitted = True


    ### Single Protein Model ###
    def generate_sp_model(self, dev_df, iteration):
        """
        choose RAPS and generate single protein (RAP) models for a given subset of dev_df
        """
        #split development data randomly to train / test
        [iteration_train_df, iteration_test_df] = train_test_split(dev_df, train_size=self.sp_train_ratio, stratify=1-dev_df['Y'])
        logger.info(f'iteration {iteration}: splitted data to train / test dataframes')
        #select relevant proteins (based on KS test) for the specific partition
        iteration_ks_proteins = self.evaluate_ks_stat(iteration_train_df)
        logger.debug(f'iteration {iteration}: ks_proteins = {iteration_ks_proteins["ks_proteins"]}')
        logger.debug(f'iteration {iteration}: bh_pval = {iteration_ks_proteins["corrected_pval"]}')
        #build single proteins model for the specific partition and selected proteins
        iteration_models = self.generate_sp_models(iteration_train_df, iteration_test_df, iteration_ks_proteins['ks_proteins'])
        self.sp_models.append({
            'ks_proteins': iteration_ks_proteins['ks_proteins'], 'proteins_stat': iteration_ks_proteins['proteins_stat'],
            'xgb_models': iteration_models, 'corrected_pval': iteration_ks_proteins['corrected_pval']})
        
        
    def evaluate_ks_stat(self, train_df):
        """
        evaluate p_val that R and NRs follows same distribution per protein (on train set), using KS test
        """
        #evaluate pval for all relevant proteins
        proteins_stat = pd.DataFrame(np.ones(len(self.proteins)), index=self.proteins, columns=['ks_pval'])
        y_train = np.array(train_df['Y'])
        for protein in self.proteins:
            if 'spearr' not in self.protein_selection_method and 'cox' not in self.protein_selection_method:
                proteins_stat.loc[protein] = _calculate_ks_pval(train_df[protein], y_train, self.n_expected_outliers, self.protein_selection_method)
            elif 'cox' in self.protein_selection_method:
                proteins_stat.loc[protein] = _calculate_cox_pval(train_df, protein, self.n_expected_outliers, self.protein_selection_method)
            else:
                proteins_stat.loc[protein] = _calculate_spearr_pval(train_df, protein, self.n_expected_outliers, self.protein_selection_method)
        #set pval threshold and select relevant features
        proteins_stat['ks_rank'] = np.argsort(np.argsort(proteins_stat['ks_pval'].values))
        proteins_stat['multiple_test_reject'], proteins_stat['corrected_pval'], _, _ = multipletests(proteins_stat['ks_pval'], method=self.multipletests_method, alpha=self.multipletests_alpha)
        #select proteins with the lowest p_value
        
        if self.number_of_KS_proteins > 1: # Fixed number of raps per iteration
            rank_col = 'ks_rank'
        else: # Choose by FDR value which its value is between (0,1)
            rank_col = 'corrected_pval'
        try:
            # print(1)
            # print(self.n_first_ks)
            # print(rank_col)
            # print(proteins_stat.sort_values(rank_col).iloc[(self.n_first_ks-1):(self.n_first_ks+self.number_of_KS_proteins-1)])
            # print(proteins_stat.sort_values(rank_col).iloc[(self.n_first_ks-1):(self.n_first_ks+self.number_of_KS_proteins-1)].index)
            ks_proteins = list(proteins_stat.sort_values(rank_col).iloc[(self.n_first_ks-1):(self.n_first_ks+self.number_of_KS_proteins-1)].index)
            # print(proteins_stat.sorted(rank_col).iloc[(self.n_first_ks-1):(self.n_first_ks+self.number_of_KS_proteins-1)])
        except:
            ks_proteins = list(proteins_stat.loc[proteins_stat[rank_col] < self.number_of_KS_proteins].index)
            print('except evaluate ks')
        if len(ks_proteins) == 0:
            raise ValueError(f'No proteins pass the required FDR {self.number_of_KS_proteins}.\nConsider enlarging its value or using a different development set to resolve this error.')
        ks_proteins.sort(key = lambda protein: proteins_stat.loc[protein, 'ks_rank'])
        #update rap counts
        for protein in ks_proteins:
            if protein in self.RAPs.keys():
                self.RAPs[protein] += 1
            else:
                self.RAPs[protein] = 1
        # calculate adjusted p-value
        
        # corrected_pval = proteins_stat.loc[proteins_stat['ks_rank'] == self.number_of_KS_proteins-1, 'corrected_pval'][0]
        corrected_pval = proteins_stat.loc[proteins_stat['ks_rank'] == len(ks_proteins)-1, 'corrected_pval'][0]
        
        return {'ks_proteins': ks_proteins, 'proteins_stat': proteins_stat, 'corrected_pval': corrected_pval}
   
    
    def generate_sp_models(self, train_df, test_df, proteins):
        """
        generate single protein models
        """
        models = {protein: [] for protein in proteins}
        test_ids = list(test_df.index)
        iteration_predictions = pd.DataFrame(np.ones((len(test_ids), len(proteins))) * np.nan, index=test_ids, columns=proteins)
        for i, protein in enumerate(proteins):            
            targets = ['Y', protein] + self.sp_features
            train_test_dict = {'train': extract_xy_arrays(targets, train_df), 
                           'test': extract_xy_arrays(targets, test_df)}
            if self.model_type == 'xgb':
                y_pred, model = self.train_xgb_model(train_test_dict, self.sp_xgb_params, self.sp_xgb_num_round)
            else:
                y_pred, model = self.train_simple_model(train_test_dict)
            iteration_predictions[protein] = y_pred
            models[protein] = model
        for patient in test_ids:
            self.prediction.loc[patient, 'y_pred_sp'].append((iteration_predictions.loc[patient] > self.sp_p_threshold).sum()) #/ len(proteins)) # Added a division by number of proteins, becuase if using FDR, the number of proteins might vary between iterations
            self.prediction.loc[patient, 'y_pred_sp_mean'].append((iteration_predictions.loc[patient]).mean())
            self.prediction.loc[patient, 'y_pred_sp_median'].append((iteration_predictions.loc[patient]).median())
        self.iteration_predictions.append(iteration_predictions)
        
        return models
    
        
    def generate_clinical_model(self, dev_df):
        """
        generate a model based on RAP model prediction and selected clinical parameters
        """
        clinical_model_df = self.extract_clinical_model_df(dev_df, self.prediction, include_Y=True)
        targets = ['Y', 'rap_score'] + self.clinical_features
        for i in range(self.cm_ensemble_size):
            [iteration_train_df, iteration_test_df] = train_test_split(clinical_model_df, train_size=self.cm_train_ratio, stratify=1-clinical_model_df['Y'])
            test_ids = list(iteration_test_df.index)
            train_test_dict = {'train': extract_xy_arrays(targets, iteration_train_df), 
                               'test': extract_xy_arrays(targets, iteration_test_df)}
            y_pred, model = self.train_xgb_model(train_test_dict, self.cm_xgb_params, self.cm_xgb_num_round)
            self.clinical_models.append(model)
            for patient, y_p in zip(test_ids, y_pred):
                self.prediction.loc[patient, 'y_pred_cm'].append(y_p)
            
            
    def average_prediction(self, label):
        """
        extract mean prediction value (either single protein or clinical model) per patient over the ensemble
        """
        for patient in self.prediction.index:
            self.prediction.loc[patient, label] = np.mean(self.prediction.loc[patient, label])
        self.prediction[label] = self.prediction[label].astype(float)
        
    
    def extract_sp_scaling_factors(self, metric='y_pred_sp'):
        """ scaling y_pred column from a range of 0-number_of_KS_proteins to a range of 0-1.
        Scaling is performed using the best method from two options, both try to best fir the observed response probability.
        The method which yields the largest range is used """
        if self.sp_scaling_method == 'Failed':
            return
        predefined_method = self.sp_scaling_method
        scaled1, factor1 = scaling_factors_score_bins(self.prediction.y, self.prediction[metric], self.sp_scaling_points)
        scaled2, factor2 = scaling_factors_population_bins(self.prediction.y, self.prediction[metric], self.sp_scaling_points)
        
        if np.isnan(factor1['slope']) & np.isnan(factor2['slope']):
            logger.warning('sp scaling failed')
            self.sp_scaling_method = 'failed'
            scaled = scaled1
            factor = factor1
        elif np.isnan(factor1['slope']):
            logger.info('sp score scaling failed. Using population scaling')
            self.sp_scaling_method = 'population'
            scaled=scaled2
            factor=factor2
        elif np.isnan(factor2['slope']):
            logger.info('sp population scaling failed. Using score scaling')
            self.sp_scaling_method = 'score'
            scaled=scaled1
            factor=factor1        
        else:
            rng1=np.nanmax(scaled1)-np.nanmin(scaled1)
            rng2=np.nanmax(scaled2)-np.nanmin(scaled2)
            if (predefined_method == 'largest_span' and rng1>rng2) or predefined_method == 'score':
                logger.info('sp score scaling selected')
                self.sp_scaling_method = 'score'
                scaled=scaled1
                factor=factor1
            elif (predefined_method == 'largest_span' and rng1<=rng2) or predefined_method == 'population':
                logger.info('sp population scaling selected')
                self.sp_scaling_method = 'population'
                scaled=scaled2
                factor=factor2
            else:
                raise ValueError('Invalid scaling method')
            
        ### set min and max to 0 an 1 and save results
        if self.sp_scaling_method != 'Failed':
            if metric == 'y_pred_sp':
                self.sp_scale_factor = factor
            elif metric == 'y_pred_sp_mean':
                self.sp_mean_scale_factor = factor
            scaled[scaled > 1] = 1
            scaled[scaled < 0] = 0
            self.prediction[metric+'_scaled'] = scaled
        print(f'****Slope and intercept: {factor}******')
       
    
    #####PREDICT#####
    def predict(self, val_df):
        """
        generate response prediction based on trained model (to be used only after fitting the model)
        returns a dataframe that includes for each patient:
            y_pred_sp - the RAP score. This is the average number of raps above thershold over all the iterations (a number in the range 0-number_of_KS_proteins)
            y_pred_sp_mean - average rap prediction over all proteins in each iteration over all iterations (a number in the range 0-1)
            y_pred_sp_scaled - the value of y_pred_sp scaled 
            RAPs - list of RAPs above threshold per patient (if RAP appers in more than one iteration, it will apper more than one time in the list)
        """

        self.eval_val_df(val_df)
        prediction = self.extract_sp_prediction(val_df)
        if len(self.clinical_features) > 0:
            prediction = self.extract_cm_prediction(val_df, prediction)
        else:
            prediction['y_pred_cm'] = prediction['y_pred_sp'].copy()
        if self.sp_scaling_method != 'Failed':
            scaled = self.sp_scale_factor['slope'] * prediction.y_pred_sp + self.sp_scale_factor['intercept']
            scaled[scaled > 1] = 1
            scaled[scaled < 0] = 0
            prediction.y_pred_sp_scaled = scaled
            scaled_mean = self.sp_mean_scale_factor['slope'] * prediction.y_pred_sp_mean + self.sp_mean_scale_factor['intercept']
            scaled_mean[scaled_mean > 1] = 1
            scaled_mean[scaled_mean < 0] = 0
            prediction.y_pred_sp_mean_scaled = scaled_mean
            
        
        return prediction
    
    
    def extract_sp_prediction(self, val_df):
        """ run the single protein (sp) model and generate some of the prediction values for each patient.
        returns a dataframe that includes for each patient:
            y_pred_sp - the RAP score. This is the average number of raps above thershold over all the iterations (a number in the range 0-number_of_KS_proteins)
            y_pred_sp_mean - average rap prediction over all proteins in each iteration over all iterations (a number in the range 0-1)
            RAPs - list of RAPs above threshold per patient (if RAP appers in more than one iteration, it will apper more than one time in the list)"""

        test_ids = list(val_df.index)
        sp_prediction = {patient: {'y_pred_sp': [], 'y_pred_sp_mean': [], 'y_pred_sp_median': [], 'y_pred_sp_scaled': np.nan, 'y_pred_sp_mean_scaled': np.nan, 'y_pred_cm': np.nan, 'RAPs': []} for patient in test_ids}
        sp_prediction = pd.DataFrame.from_dict(sp_prediction, orient='index')
        for sp_model in self.sp_models:
            proteins = sp_model['ks_proteins']
            sp_pred = np.zeros([len(test_ids), len(proteins)])
            for i, protein in enumerate(proteins):
                targets = [protein] + self.sp_features
                val_dict = extract_xy_arrays(targets, val_df)
                try:
                    if self.model_type == 'nb':
                        sp_pred[:, i] = np.array([p_vec[1] for p_vec in sp_model['xgb_models'][protein].predict_proba(val_dict['X'])])
                    elif self.model_type != 'xgb':
                        sp_pred[:, i] = sp_model['xgb_models'][protein].predict(val_dict['X'])
                    else:
                        dval = xgb.DMatrix(val_dict['X'])
                        sp_pred[:, i] = sp_model['xgb_models'][protein].predict(dval)
                except:
                    dval = xgb.DMatrix(val_dict['X'])
                    sp_pred[:, i] = sp_model['xgb_models'][protein].predict(dval)
                for patient, pred in zip(test_ids, sp_pred[:, i]):
                    if pred > self.sp_p_threshold:
                        sp_prediction.loc[patient, 'RAPs'].append(protein)
            iteration_pred = np.sum(sp_pred > self.sp_p_threshold, axis=1) # / sp_pred.shape[1] TODO: remove or keep. removed divide for debugging.
            iteration_pred_mean = np.mean(sp_pred, axis=1)
            iteration_pred_median = np.median(sp_pred, axis=1)
            for patient, y_p, y_p_mean, y_p_median in zip(test_ids, iteration_pred, iteration_pred_mean, iteration_pred_median):
                sp_prediction.loc[patient, 'y_pred_sp'].append(y_p)
                sp_prediction.loc[patient, 'y_pred_sp_mean'].append(y_p_mean)
                sp_prediction.loc[patient, 'y_pred_sp_median'].append(y_p_median)
        for patient in test_ids:
            sp_prediction.loc[patient, 'y_pred_sp'] = np.mean(sp_prediction.loc[patient, 'y_pred_sp'])
            sp_prediction.loc[patient, 'y_pred_sp_mean'] = np.mean(sp_prediction.loc[patient, 'y_pred_sp_mean'])
            sp_prediction.loc[patient, 'y_pred_sp_median'] = np.mean(sp_prediction.loc[patient, 'y_pred_sp_median'])
        sp_prediction['y_pred_sp'] = sp_prediction['y_pred_sp'].astype(float)
        sp_prediction['y_pred_sp_mean'] = sp_prediction['y_pred_sp_mean'].astype(float)
        sp_prediction['y_pred_sp_median'] = sp_prediction['y_pred_sp_median'].astype(float)
            
        return sp_prediction
    
    
    def extract_cm_prediction(self, val_df, sp_prediction):
        """ adds the clincal score (y_pred_cm) to each patient prediction"""
        val_ids = list(val_df.index)
        cm_predictions_array = np.zeros([len(val_ids), self.cm_ensemble_size])
        clinical_model_df = self.extract_clinical_model_df(val_df, sp_prediction, include_Y=False)
        targets = ['rap_score'] + self.clinical_features
        val_dict = extract_xy_arrays(targets, clinical_model_df)
        dval = xgb.DMatrix(val_dict['X'])
        for i, cm_model in enumerate(self.clinical_models):
            cm_predictions_array[:, i] = cm_model.predict(dval)
        cm_prediction = sp_prediction.copy()
        cm_prediction.loc[val_ids, 'y_pred_cm'] = np.mean(cm_predictions_array, axis=1)
        
        return cm_prediction
    

    #####  SUPPORT METHODS #####
    def train_xgb_model(self, train_test_dict, xgb_params, xgb_num_round):
        dtrain = xgb.DMatrix(train_test_dict['train']['X'], train_test_dict['train']['y'])
        ypred = 0
        model = xgb.train(xgb_params, dtrain, xgb_num_round)
        if train_test_dict['test']['y'].size > 0:
            dtest = xgb.DMatrix(train_test_dict['test']['X'])
            ypred = model.predict(dtest)
        
        return ypred, model
    
    def train_simple_model(self, train_test_dict):
        X_train = train_test_dict['train']['X']
        y_train = train_test_dict['train']['y']
        ypred = 0
        if self.model_type == 'logreg':
            model = LogisticRegression()
        elif self.model_type == 'nb':
            model = GaussianNB()
        model.fit(X_train, y_train) # to scale X?
        if train_test_dict['test']['y'].size > 0:
            X_test = train_test_dict['test']['X']
            if self.model_type == 'nb':
                ypred = np.array([p_vec[1] for p_vec in model.predict_proba(X_test)])
            else:
                ypred = model.predict(X_test)
        
        return ypred, model
    
    
    def extract_clinical_model_df(self, df, prediction, include_Y):
        if include_Y:
            relevant_columns = self.clinical_features + ['Y']
        else:
            relevant_columns = self.clinical_features
        clinical_model_df = df[relevant_columns]
        rap_score = prediction.loc[clinical_model_df.index, 'y_pred_sp'].copy()
        rap_score.name = 'rap_score'
        clinical_model_df = clinical_model_df.join(rap_score, how='inner')
        
        return clinical_model_df
    

    def evaluate_auc(self):
        y = np.array(self.prediction.y)
        y_sp = np.array(self.prediction.y_pred_sp)
        y_sp_mean = np.array(self.prediction.y_pred_sp_mean)
        y_sp_median = np.array(self.prediction.y_pred_sp_median)
        y_cm = np.array(self.prediction.y_pred_cm)
        ind = np.isfinite(y_sp)
        self.sp_auc = roc_auc_score(y[ind].astype(int), y_sp[ind])
        self.sp_mean_auc = roc_auc_score(y[ind].astype(int), y_sp_mean[ind])
        self.sp_median_auc = roc_auc_score(y[ind].astype(int), y_sp_median[ind])
        if len(self.clinical_features) > 0:
            ind = np.isfinite(y_cm)    
            self.cm_auc = roc_auc_score(y[ind].astype(int), y_cm[ind])
        
        
    ##### CLASS TESTS #####
    def eval_initialized_values(self):        
        if len(self.proteins) == 0 or type(self.proteins) not in [list, np.ndarray]:
            raise AssertionError('protein features not specified')
        if not (0 < self.number_of_KS_proteins < 5000):
            raise AssertionError(f'number_of_KS_proteins = {self.number_of_KS_proteins} is not a valid value, allowed values are in range [1...4999]')
        if (len(self.proteins) < self.number_of_KS_proteins) and (self.number_of_KS_proteins > 1):
            raise AssertionError(f'Number of protein columns ({len(self.proteins)}) must be greater or equal to the number_of_KS_proteins ({self.number_of_KS_proteins})')
        if np.logical_xor(len(self.clinical_features) == 0, self.cm_ensemble_size == 0):
            raise AssertionError(f'number of clinical features = {len(self.clinical_features)} while cm_ensemble_size = {self.cm_ensemble_size}')
        valid_scaling_methods = ['largest_span', 'score', 'population']
        if self.sp_scaling_method not in valid_scaling_methods:
            raise AssertionError(f'{self.sp_scaling_method} is not a valid scaling method. Please choose from {", ".join(valid_scaling_methods)}')
        # check validity of sp_p_threshold
        is_valid_sp_p_thr = False
        if type(self.sp_p_threshold) == str:
            if self.sp_p_threshold in ['mean_response_rate']:
                is_valid_sp_p_thr = True
        if type(self.sp_p_threshold) == float:
            if (0 < self.sp_p_threshold < 1):
                is_valid_sp_p_thr = True
        if not is_valid_sp_p_thr:
            raise AssertionError(f'sp_p_threshold = {self.sp_p_threshold} is not a valid value, allowed values are "mean_response_rate" or floats in range(0,1).')
            
    def eval_dev_df(self, train_df):
        if self.is_fitted:
            raise AssertionError('RapResponePredictor object was already trained. In order to train additional model initiate new object')
        required_columns = ['Y'] + self.sp_features + self.clinical_features + self.proteins
        missing_columns = [column for column in required_columns if column not in train_df.columns]
        if len(missing_columns) > 0:
                raise AssertionError(f'mandatory columns: {missing_columns} are missing from train_df columns')
        proteins_df = train_df[required_columns]
        if proteins_df.isnull().values.any():
            raise AssertionError('train_df contains NaN values. NaNs are not supported.')
        if not(all(np.unique(train_df['Y']) == np.array([0, 1]))):
            raise AssertionError('Y values must be either 0 or 1')
            
    def eval_val_df(self, test_df):
        if not self.is_fitted:
            raise AssertionError('RapResponePredictor object was not fitted. In order to predict, train the model first')
        required_columns = self.sp_features + self.clinical_features + list(self.RAPs.keys())
        missing_columns = [column for column in required_columns if column not in test_df.columns]
        if len(missing_columns) > 0:
                raise AssertionError(f'mandatory columns: {missing_columns} are missing from test_df columns')
        proteins_df = test_df[required_columns]
        if proteins_df.isnull().values.any():
            raise AssertionError('test_df contains NaN values. NaNs are not supported.')
            

##### AUXILARY FUNCTIONS #####
def _robust_t_test(group_a_vals, group_b_vals): # a guesstimated test
    med_a = np.median(group_a_vals)
    med_b = np.median(group_b_vals)
    mad_a = stats.median_abs_deviation(group_a_vals)
    mad_b = stats.median_abs_deviation(group_b_vals)
    s_delta = (mad_a**2/len(group_a_vals) + mad_b**2/len(group_b_vals)) ** 0.5
    robust_t = (abs(med_a - med_b)) / s_delta
    return robust_t, -robust_t


def _calculate_ks_pval(expression_level, label, n_outliers, protein_selection_method): #_robust_t_test stats.ttest_ind, stats.mannwhitneyu, stats.ttest_ind, stats.kstest # Gil change before: stats.kstest
    '''Returns Kolmogorov-Smirnov test (or by other metric - mw/rob_t) p-value after excluding n_outliers on
    each side of each group distribution'''
    if protein_selection_method == 'ks':
        stat_test_func = stats.kstest
    elif protein_selection_method == 'mw':
        stat_test_func = stats.mannwhitneyu
    elif protein_selection_method == 'rob_t':
        stat_test_func = _robust_t_test
    elif protein_selection_method == 'tt':
        stat_test_func = stats.ttest_ind
    else:
        raise(f'Unknown protein selection method: {protein_selection_method}')
    if n_outliers != 0:
        group_a_values = np.sort(expression_level[label==0])[n_outliers:-n_outliers]
        group_b_values = np.sort(expression_level[label==1])[n_outliers:-n_outliers]
    else:
        group_a_values = np.sort(expression_level[label==0])
        group_b_values = np.sort(expression_level[label==1])
    s, pval = stat_test_func(group_a_values, group_b_values)
    
    return pval

def _calculate_spearr_pval(train_df, protein, n_outliers, protein_selection_method): #_robust_t_test stats.ttest_ind, stats.mannwhitneyu, stats.ttest_ind, stats.kstest # Gil change before: stats.kstest
    '''Returns spearmanr test p-value after excluding n_outliers*2 on
    each side the distribution'''
    if protein_selection_method == 'spearr_os':
        corr_col = [x for x in train_df.columns if 'os' in x.lower() and 'duration' in x.lower() and 'cens' not in x.lower()][0]
    elif protein_selection_method == 'spearr_pfs':
        corr_col = [x for x in train_df.columns if 'pfs' in x.lower() and 'duration' in x.lower() and 'cens' not in x.lower()][0]
    else:
        raise(f'Unknown protein selection method: {protein_selection_method}')
    df_corr = train_df[[protein, corr_col]].copy().dropna()
    df_corr = df_corr.sort_values(by=protein)
    df_corr = df_corr.iloc[int(n_outliers*2):-int(n_outliers*2)] # We remove twice the number of outliers, since in binary methods we remove from both R and NR
        
    _, pval = stats.spearmanr(df_corr[protein], df_corr[corr_col])
    
    return pval

def _calculate_cox_pval(train_df, protein, n_outliers, protein_selection_method, spearr_th=0.2): #_robust_t_test stats.ttest_ind, stats.mannwhitneyu, stats.ttest_ind, stats.kstest # Gil change before: stats.kstest
    '''Returns Cox regression p-value after excluding n_outliers*2 on
    each side the distribution. Only for proteins with spearman r higher than the threshold.'''
    if protein_selection_method == 'cox_os':
        target = 'os'
    elif protein_selection_method == 'cox_pfs':
        target = 'pfs'
    else:
        raise(f'Unknown protein selection method: {protein_selection_method}')
    duration_col = [x for x in train_df.columns if target in x.lower() and 'duration' in x.lower() and 'cens' not in x.lower()][0]
    event_col = [x for x in train_df.columns if target in x.lower() and 'event' in x.lower() and 'cens' not in x.lower()][0]
    
    df_cox = train_df[[protein, duration_col, event_col]].copy().dropna()
    df_cox = df_cox.sort_values(by=protein)
    df_cox = df_cox.iloc[int(n_outliers*2):-int(n_outliers*2)] # We remove twice the number of outliers, since in binary methods we remove from both R and NR
    spear_r, p_val = stats.spearmanr(df_cox[protein], df_cox[duration_col])
    if np.abs(spear_r) < spearr_th:
        return p_val + 0.5 # if did not pass the threshold return with high p_value.
    cph = lifelines.CoxPHFitter()
    cph.fit(df_cox, duration_col=duration_col, event_col=event_col)#, fit_options={"max_steps": 5, "precision": 1E-3})
    z, p_val = cph.summary.z.iloc[0], cph.summary.p.iloc[0]
    # print(58)
        
    return p_val



# def _calculate_ks_pval(expression_level, label, n_outliers): # original
#     '''Returns Kolmogorov-Smirnov test p-value after excluding n_outliers on
#     each side of each group distribution'''
#     group_a_values = np.sort(expression_level[label==0])[n_outliers:-n_outliers]
#     group_b_values = np.sort(expression_level[label==1])[n_outliers:-n_outliers]
#     s, pval = stats.kstest(group_a_values, group_b_values)
    
#     return pval


def scaling_factors_score_bins(y, y_pred, scaling_points, reg_type='ransac'):
    '''calculate scaling factors by equal sized score bins'''
    y = np.array(y)
    y_pred = np.array(y_pred)
    interval_size = (np.nanmax(y_pred) - np.nanmin(y_pred)) / (scaling_points + 1)
    y_min = np.nanmin(y_pred) + interval_size / 2
    y_max = np.nanmax(y_pred) - interval_size / 2
    yp_vec = np.linspace(y_min, y_max, scaling_points)
    yp_scaled = np.zeros(yp_vec.size)
    for i, yp in enumerate(yp_vec):
        ind = np.logical_and(y_pred >= yp - interval_size / 2, y_pred <= yp + interval_size / 2)
        yp_scaled[i] = np.mean(y[ind])
    empty_bins = np.isnan(yp_scaled)
    
    slope, intercept = get_linear_regression_params(x=yp_scaled[~empty_bins], y=yp_vec[~empty_bins], reg_type=reg_type)
    scale_factor = {'slope': slope, 'intercept': intercept}
    if scale_factor['slope'] <= 0:
        y_pred_scaled = np.nan * np.ones(len(y_pred))
        scale_factor = {'slope': np.nan, 'intercept': np.nan}
        return y_pred_scaled, scale_factor
    y_pred_scaled = scale_factor['slope'] * y_pred + scale_factor['intercept']
    return y_pred_scaled, scale_factor
    
    
def scaling_factors_population_bins(y, y_pred, scaling_points, reg_type='ransac'):
    ''' calculate scaling factors using bins with equal sized population '''
    y = np.array(y)
    y_pred = np.array(y_pred)
    nan_preds = np.isnan(y_pred)
    try:
        intervals = list(pd.qcut(y_pred[~nan_preds], scaling_points).categories)
    except ValueError:
        y_pred_scaled = np.nan * np.ones(len(y_pred))
        scale_factor = {'slope': np.nan, 'intercept': np.nan}
        return y_pred_scaled, scale_factor
    yp_vec = np.zeros(len(intervals))
    yp_scaled = np.zeros(len(intervals))
    for i, yp in enumerate(intervals):
        yp_vec[i] = (yp.left + yp.right) / 2
        ind = np.logical_and(y_pred > yp.left, y_pred <= yp.right)
        yp_scaled[i] = np.mean(y[ind])
    empty_bins = np.isnan(yp_scaled)
    
    
    slope, intercept = get_linear_regression_params(x=yp_scaled[~empty_bins], y=yp_vec[~empty_bins], reg_type=reg_type)
    scale_factor = {'slope': slope, 'intercept': intercept}
    if scale_factor['slope'] <= 0:
        y_pred_scaled = np.nan * np.ones(len(y_pred))
        scale_factor = {'slope': np.nan, 'intercept': np.nan}
        return y_pred_scaled, scale_factor
    y_pred_scaled = scale_factor['slope'] * y_pred + scale_factor['intercept']
    return y_pred_scaled, scale_factor


def get_linear_regression_params(x, y, reg_type='ransac'):
    if reg_type == 'lsq':
        a = np.polyfit(y, x, 1)
        slope = a[0]
        intercept = a[1]
    elif reg_type == 'theil-sen':
        slope, intercept, _, _ = stats.theilslopes(x, y) # stats.theilslopes(y, x)
    elif reg_type == 'ransac':
        ransac = RANSACRegressor()
        ransac.fit(y.reshape(-1, 1), x.reshape(-1, 1))
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
    else:
        print(f'Warning: {reg_type} regression is not implemented.')
        slope = np.nan
        intercept = np.nan
        print(f'*****slope: {slope}; intercept: {intercept}*****')
    return slope, intercept
        
    


# def get_huber_loss(params, x, y):
#     # Huber regression loss
#     slope, intercept = params
#     y_pred = slope * x + intercept
#     residuals = y - y_pred
#     huber_loss = np.sum(np.where(np.abs(residuals) < 1, 0.5 * residuals**2, np.abs(residuals) - 0.5))
#     return huber_loss