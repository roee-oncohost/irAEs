##### IMPORTS #####
import numpy as np
import pandas as pd
import pickle
from logger_manager import getLogger
from copy import deepcopy

logger = getLogger('data_parser')

def prepare_v2_0(xy_df, sets, log_flag, last_col, start_col=53):
    xy_df = deepcopy(xy_df)
    try:
        if sum(xy_df.columns.isin(['PDL1_less_than_1', 'PDL1_1_49', 'PDL1_more_50'])) < 3:
            xy_df['PDL1_less_than_1'] = xy_df['PDLLevel'] == 'Negative'
            xy_df['PDL1_1_49'] = xy_df['PDLLevel'] == 'Low'
            xy_df['PDL1_more_50'] = xy_df['PDLLevel'] == 'High'
        if sum(xy_df.columns.isin(['ORR1Reported', 'ORR2Reported', 'OneYearDCBCalculated'])) >0:#== 3:#ori changed
            xy_df.rename({'ORR1Reported': 'ORR1_early_death', 'ORR2Reported': 'ORR2_early_death', 'OneYearDCBCalculated': 'DCB at 1 year',
                          'OSEvent': 'OS Event', 'PFSEvent': 'PFS Event', 'OSDuration': 'OS Duration', 'OSCensoredDuration': 'OS Duration', 'PFSDuration': 'PFS Duration', 'PFSCensoredDuration': 'PFS Duration'}, axis=1, inplace=True)
        xy_df['OS Event'] = xy_df['OS Event'].map({'None': 0, 'Death': 1})
        xy_df['PFS Event'] = xy_df['PFS Event'].map({'None': 0, 'Progression': 1})
        xy_df['ORR1_early_death'].fillna('Unknown')
        xy_df['ORR2_early_death'].fillna('Unknown')
        xy_df['Set'] = xy_df[sets]
    except:
        pass
    if log_flag:
        xy_df.iloc[:,start_col:last_col] = xy_df.iloc[:,start_col:last_col].apply(np.log2)  #dcb mode
    return xy_df



# def prepare_v2_0(xy_df):
#     xy_df = deepcopy(xy_df)
#     if sum(xy_df.columns.isin(['PDL1_less_than_1', 'PDL1_1_49', 'PDL1_more_50'])) < 3:
#         xy_df['PDL1_less_than_1'] = xy_df['PD-L1 Level'] == 'Negative'
#         xy_df['PDL1_1_49'] = xy_df['PD-L1 Level'] == 'Low'
#         xy_df['PDL1_more_50'] = xy_df['PD-L1 Level'] == 'High'
#     if sum(xy_df.columns.isin(['ORR1Reported', 'ORR2Reported', 'OneYearDCBCalculated'])) == 3:
#         xy_df.rename({'ORR1Reported': 'ORR1_early_death', 'ORR2Reported': 'ORR2_early_death', 'OneYearDCBCalculated': 'DCB at 1 year',
#                       'OSEvent': 'OS Event', 'PFSEvent': 'PFS Event', 'OSDuration': 'OS Duration', 'PFSDuration': 'PFS Duration'}, axis=1, inplace=True)    
#     xy_df['ORR1_early_death'].fillna('Unknown')
#     xy_df['ORR2_early_death'].fillna('Unknown')
#     xy_df['OS Event'] = xy_df['OS Event'].map({'None': 0, 'Death': 1})
#     xy_df['PFS Event'] = xy_df['PFS Event'].map({'None': 0, 'Progression': 1})
#     xy_df['Set'] = xy_df['SetV2.0a'].copy()
#     include = (~xy_df['SetV2.0a'].isna()) & (xy_df['ExcludeV2.0'] == False) & (xy_df['Indication'] == 'NSCLC')  & (xy_df['Line'] == 'First') & (xy_df['TreatmentCombo'].isin(['ICI', 'ICI + Chemo']))
#     if sum(~include) > 0:
#         logger.info(f'Excluding {sum(~include)} patients based on clinical paramters (Advanced line\\non-NSCLC\\non-ICI patients\\ExcludeV2.0 column. {sum(include)} patients remaining')
#     xy_df = xy_df.loc[include]
#     return xy_df


def parse_x(xy_df):
    """
    prepare xy dataframe clinical data: set line, sex and pd-l1 conventions.
    input dataframe should have columns and values:
        'Sex': ['Male', 'Female']
        'Line': ['First', 'Advanced']
        'PDL1_less_than_1' : [0, 1]
        'PDL1_1_49' : [0, 1]
        'PDL1_more_50' : [0, 1]
    SubjectIds should be given as the dataframe index (format: CN-CEN-SUBJ-INDICATION, e.g., DE-041-1234-NSCLC)
        
    Parameters
    ----------
    xy_df : DataFrame
        xy_df should have columns and values:
            'Sex': ['Male', 'Female']
            'Line': ['First', 'Advanced']
            'PDL1_less_than_1' : [0, 1]
            'PDL1_1_49' : [0, 1]
            'PDL1_more_50' : [0, 1]
    
    Returns
    -------
    DataFrame :
        A copy of xy_df with sex, line columns converted to binary label.
        A new column named PDL1 with numeric values between -1 and 2
        A new column with medical center parsed from input dataframe index
    
    Examples
    --------
    
    >>> dat = pd.DataFrame.from_dict({'Sex': ['Male', 'Female', 'Male', 'Female'],
                                     'Line': ['First', 'Advanced', 'Advanced', 'First'],
                                     'PDL1_less_than_1': [0, 1, 0, 0],
                                     'PDL1_1_49': [0, 0, 0, 1],
                                     'PDL1_more_50': [1, 0, 0, 0],
                                     'Set': ['Dev', 'Dev', 'Val', 'Val'],
                                     'Response': ['NR', 'SD', 'Unknown', 'R']})
    >>> parse_x(dat)
    
        Sex  Line  PDL1_less_than_1  PDL1_1_49  PDL1_more_50  Set Response  PDL1
     0    0     0                 0          0             1  Dev       NR     2
     1    1     1                 1          0             0  Dev        R     0
     2    0     1                 0          0             0  Val  Unknown    -1
     3    1     0                 0          1             0  Val        R     1
    """
    xy_df = deepcopy(xy_df) # copy prevents the function from editing the input dataframe
    # adjust Sex, Line notations
    xy_df = xy_df.replace({'Sex': {'Male': 0, 'Female': 1},
                           'Line': {'First': 0, 'Advanced': 1}})
    # add adjusted pdl1 column
    xy_df['PDL1'] = -1 + 1 * xy_df['PDL1_less_than_1'] + 2 * xy_df['PDL1_1_49'] + 3 * xy_df['PDL1_more_50']
    # add medical center id column
    if type(xy_df.index) != pd.core.indexes.base.Index:
        logger.warning('xy_df should have SubjectIds as index (format: CN-CEN-SUBJ-INDICATION, e.g., DE-041-1234-NSCLC)')
    else:
        xy_df['center_id'] = xy_df.index.str.slice(3,6)
    
    return xy_df


def parse_y(xy_df, base_col, new_col='Y', y_map={'R': 1, 'NR': 0, 'SD': 0.5}, drop_unknown_y=False,
              set_col='Set', dev_flag='Dev', val_flag='Val'):
    """
    adds a new y column with label changed to binary.
    split dataframe to two dataframes: devlopment dataframe and validation datafram according set_col
    
    xy_df : DataFrame
    
    base_col : str
        A name of a column in xy_df that will be converted to a binary label.
        
    new_col : str
        A name for the new y column to be added to xy_df.
        
    y_map : dict
        A dictionary applying a binary label to each value in xy_df[base_col]
        
    drop_unknown_y : bool
        If True, rows with no valid y_map key will be deleted.
        If False, rows with no valid y_map key will contain nan values.
        
    set_col : str
        Column name in xy_df containing the division to two sets
        
    dev_flag : str
        The value in set_col that directs a sample to the development set
        
    val_flag : str
        The value in set_col that directs a sample to the validation set

    Returns
    -------
    dev_df : DataFrame
        A dataframe with all development set samples and a new binary y column
    
    val_df : DataFrame
        A dataframe with all validation set samples and a new binary y column
    
    Examples
    --------
    >>> dat = pd.DataFrame.from_dict({'Sex': ['Male', 'Female', 'Male', 'Female'],
                                     'Line': ['First', 'Advanced', 'Advanced', 'First'],
                                     'PDL1_less_than_1': [0, 1, 0, 0],
                                     'PDL1_1_49': [0, 0, 0, 1],
                                     'PDL1_more_50': [1, 0, 0, 0],
                                     'Set': ['Dev', 'Dev', 'Val', 'Val'],
                                     'Response': ['NR', 'SD', 'Unknown', 'R']})
    >>> dev_df, val_df = parse_y(dat, base_col='Response', y_map={'R': 1, 'NR': 0, 'SD': 1})
    [WARNING] There are 1 lines with unknown y values. Use drop_unknown_y=True to exclude them.
    [INFO] data_parser: Found 2 train lines and 2 test lines
    
    >>> print(dev_df)
    
          Sex      Line  PDL1_less_than_1  PDL1_1_49  PDL1_more_50  Set Response  Y
    0    Male     First                 0          0             1  Dev       NR  0
    1  Female  Advanced                 1          0             0  Dev       SD  1
    """
    xy_df = deepcopy(xy_df)
    # generate new y column
    xy_df[new_col] = xy_df[base_col]
    # map values in y column according to y_map
    xy_df = xy_df.replace({new_col: y_map})
    include = xy_df[base_col].isin(y_map.keys())
    n_exclude = (~include).sum()
    if drop_unknown_y:
        # remove lines with unknown or unrequested response
        if n_exclude > 0:
            logger.info(f'Excluding {n_exclude} lines with unknown response (remaining {len(xy_df)-n_exclude} of {len(xy_df)} lines)')
        xy_df = xy_df[include]
    else:
        if n_exclude > 0:
            logger.warning(f'There are {n_exclude} lines with unknown y values. Use drop_unknown_y=True to exclude them.')
    
    # split to dev / val df
    dev_df = xy_df[xy_df[set_col] == dev_flag]
    val_df = xy_df[xy_df[set_col] == val_flag]
    
    if len(dev_df) + len(val_df) < len(xy_df):
        raise ValueError('Some rows were not flagged to either development of validation set, all rows in the dataframe must have a valid flag.')
        
    logger.info(f'Found {len(dev_df)} development set lines and {len(val_df)} validation set lines')
    
    return dev_df, val_df
        

def extract_proteins_list(features, horizon):
    """
    extract list of proteins out of train_df columns
    """
    features = list(features)
    t0_proteins = [f for f in features if f.endswith('_T0')]
    t1_proteins = [f for f in features if f.endswith('_T1')]
    if horizon == 'T0':
        proteins = t0_proteins
    elif horizon == 'T1':
        proteins = t1_proteins
    elif horizon is None:
        proteins = t0_proteins + t1_proteins
    return proteins
 

def extract_xy_arrays(targets, xy_df):
    if 'Y' in targets:
        y = np.array(xy_df['Y']).astype(int)
    else:
        y = np.nan
    X_features = [target for target in targets if target != 'Y']
    X = xy_df[X_features]
    
    return {'X': X, 'y': y}


def restrict_centers(proteomic_df, center_id, exclude=True):
    """
    remove patients from a specific center.
    center is denoted by the digits str (e.g. Sheba = '006')
    """
    if exclude:
        new_proteomic_df = proteomic_df[proteomic_df['SubjectId'].str.split('-').str[1] != center_id]
    else:
        new_proteomic_df = proteomic_df[proteomic_df['SubjectId'].str.split('-').str[1] == center_id]
    
    return new_proteomic_df
         
 
def save_model(model, output_file):
    """
    save model to a file (pickle wrapper)
    """
    with open(output_file,'wb') as f:
        pickle.dump(model, f)
        
        
def load_model(input_file):
    """
    load model from a file (pickle wrapper)
    """
    return pickle.load(open(input_file, 'rb'))
 