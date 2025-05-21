import pickle
import numpy as np
import pandas as pd
import seaborn as sns

PROPHET_MODEL_FILE = r'models\DCB.pkl'
VALIDATION_SET_FILE = r'C:\Git\CL_private\2023_11 - irAE classifier\data\irAE dataset with new model predictions.csv'

# Load validation set samples 
# val_df = pd.read_csv(VALIDATION_SET_FILE, index_col='SubjectId', low_memory=False)

# # Log transform protein columns, add '_T0' suffix
# first_seqid = '10000-28'
# last_seqid = '9999-1'
# proteomics_cols = val_df.columns[np.where(val_df.columns == first_seqid)[0][0]:np.where(val_df.columns == last_seqid)[0][0]]
# val_df.loc[:,proteomics_cols] = np.log2(val_df.loc[:,proteomics_cols])
# val_df.rename({col: col + '_T0' for col in proteomics_cols}, axis=1, inplace=True)
# proteomics_cols = proteomics_cols + '_T0'
# val_df['Sex'] = val_df['Gender'].map({'Male': 0, 'Female': 1})

# # Load PROphet model
# with open(PROPHET_MODEL_FILE, 'rb') as f:
#     model = pickle.load(f)

# # Run prediction for validation set (Important: make sure none of the samples were in the dev set)
# pred = model.predict(val_df)

# Convert response probability to PROphet score
def rap_score_to_prophet_score(dev_rap_scores, subj_rap_score):
    prophet_score = 10*(dev_rap_scores<subj_rap_score).mean()
    return prophet_score

# prophet_scores = pred['y_pred_sp_scaled'].apply(lambda x: rap_score_to_prophet_score(model.prediction['y_pred_sp_scaled'],x))
# prophet_result = (prophet_scores >= 5).map({False: 'Negative', True: 'Positive'})

# pred_df = pd.DataFrame(zip(pred['y_pred_sp_scaled'], prophet_scores, prophet_result),
#                        index=pred.index, columns=['Response probability', 'PROphet Score', 'PROphet result'])

# # Plot transformation result
# sns.scatterplot(pred_df, x='Response probability', y='PROphet Score', hue='PROphet result')