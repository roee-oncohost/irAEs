
'''
PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = 'Severe_AE'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = 'rap_model_sheba_(3+ AEs)'
DATA_FILE = "241020_AE_pats"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 1
LOG_FLAG = 0   
'''

'''
PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = 'Severe_AE'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = 'rap_model_sheba_Severe_AE2'
DATA_FILE = "Gastroenterology_AE_with_severity_and_CB_proteomics_sheba_filtered"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 3
LOG_FLAG = 0   
'''

'''
PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = '2+ AEs'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = f'rap_model_{filter_name}_({RESPONSE_METRIC})'
DATA_FILE = "AECount_proteomics_sheba_filtered"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 6
LOG_FLAG = 0   

'''
'''
PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = 'ClinicalBenefit'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = f'rap_model_{filter_name}_Systemic_AE'#{RESPONSE_METRIC}'#
DATA_FILE = "Systemic_AE_with_severity_and_CB_proteomics_sheba_filtered"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 3
LOG_FLAG = 0   
'''

'''
PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = 'fatigue'#'nausea|vomiting|constipation|colitis'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = f'rap_model_{filter_name}_parsedAE({RESPONSE_METRIC})'.replace('|','OR')
DATA_FILE = "Parsed_AE_proteomics_sheba_filtered"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 8
LOG_FLAG = 0   
'''


'''
PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = 'ClinicalBenefit'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = f'rap_model_{filter_name}_Gastroenterology_AE'
DATA_FILE = "Gastroenterology_AE_with_severity_and_CB_proteomics_sheba_filtered"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 3
LOG_FLAG = 0   
'''

'''
PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = 'Severe_AE'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = f'rap_model_{filter_name}_Systemic_AE'
DATA_FILE = "Systemic_AE_with_severity_proteomics_sheba_filtered"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 2 
LOG_FLAG = 0   
'''

PROTEIN_FILTER_FILE = './data/SeqId Annotations Secreted & Sheba Filters.csv'

CONFIGURATION_FILE = 'config_adverse_events_prediction.json'
RESPONSE_METRIC = 'Systemic_AE'  
SETS = 'SetAEV4'  

filter_name = 'sheba'
   
MODEL_NAME = f'rap_model_{filter_name}_{RESPONSE_METRIC}'
DATA_FILE = "Systemic_AE_with_severity_and_CB_proteomics_sheba_filtered"

FOLDER = f"./data/"  
DATA_PATH = f"{FOLDER}/{DATA_FILE}.csv"

START_COL = 3
LOG_FLAG = 0   
