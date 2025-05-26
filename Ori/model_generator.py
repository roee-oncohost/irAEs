##### IMPORTS #####
import json
import pickle
import numpy as np
import pandas as pd

#%%

from rap_response_predictor import RapResponsePredictor
from data_parser import parse_x, parse_y, extract_proteins_list

DATA_FILE = './data/230305_V2.1c_classifier_df.csv'
PROTEIN_FILTER_FILE = './data/Sheba filter.csv'  # './data/SeqId Annotations Secreted & Sheba Filters.csv'
CONFIGURATION_FILE = 'response_prediction_configuration.json'
RESPONSE_METRICS = ['DCB at 1 year']  # ['ORR1_early_death', 'ORR2_early_death', 'DCB at 1 year']
HORIZON = 'T0'
SEED = 2

#%% read protein annotations and filters file
protein_df = pd.read_csv(PROTEIN_FILTER_FILE, index_col=0, low_memory=False)
protein_filt = protein_df.index.values.tolist() #list(protein_df.loc[protein_df['Sheba p-value'] > 0.05].index)
protein_filt = [feature + '_' + HORIZON for feature in protein_filt]
protein_filt = [protein.replace('_T0','') for protein in protein_filt]

#%% read datafile
xy_df = pd.read_csv(DATA_FILE, low_memory=False, index_col='SubjectId')
#xy_df.drop(labels='Unnamed: 0', axis=1, inplace=True)
#xy_df = parse_x(xy_df)
xy_df['Sex'] = xy_df['Sex'].map({'Male':0,'Female':1})


#%% read configuration file
with open(CONFIGURATION_FILE, 'r') as f:
    conf = json.load(f)['rap_response_predictor']

#%% prepare Y column
globl = conf['globals']
y_map = {'R': globl['RESPONDER'], 'SD': globl['STABLE'], 'NR': globl['NONRESPONDER']}
response_col = 'OneYearDCBCalculated'
response_metric = RESPONSE_METRICS[0]
#for response_metric in RESPONSE_METRICS:
# parse y column and separate dev from validation set
xy_df['Y'] = xy_df[response_col].map(y_map)
dev_df = xy_df[xy_df['SetV2.0b'] == 'Dev'] #dev_df, _ = parse_y(xy_df, response_metric, y_map=y_map, drop_unknown_y=True)
params = conf[f'{response_metric}_initialization_params']
params['proteins'] = list(dev_df.columns[dev_df.columns.isin(protein_filt)])
np.random.seed(SEED)
rap_response_predictor = RapResponsePredictor(**params)
rap_response_predictor.fit(dev_df)
output_file = f'./models/models_file_v1.1e_FILTERED_full_{response_metric}_seed_{SEED}.pkl'
print(response_metric, rap_response_predictor.sp_auc)
with open(output_file, 'wb') as f:
    pickle.dump(rap_response_predictor, f)
            