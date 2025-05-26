import pandas as pd

somalogic_names = pd.read_excel('data/SomaScan_V4.1_7K_Annotated_Content_20210616.xlsx',sheet_name='Annotations',skiprows=[0,1,2,3,4,5,6,7],index_col=0)
soma_colnames = ['Target Name','Target Full Name','UniProt ID']
df_prot1 = pd.read_csv('data/AECount_proteomics_sheba_filtered.csv',low_memory=False,index_col=0)
df_prot2 = pd.read_csv('data/Systemic_AE_with_severity_and_CB_proteomics_sheba_filtered.csv',low_memory=False,index_col=0)
df_prot = pd.concat([df_prot1,df_prot2.iloc[:,:3]],axis=1)


label_names = ['(1+ AEs)','(2+ AEs)','(3+ AEs)','(4+ AEs)','(5+ AEs)','Systemic_AE','Severe_AE']
raps_all_df = pd.DataFrame(0,columns = label_names+['Any'],index = df_prot.columns[6:-4])
repeated_raps_all_list = []
for label_name in label_names:
    RAPs_filename = f'raps/rap_model_sheba_{label_name}.csv'
    lbl = label_name.replace('(','').replace(')','')

    raps = pd.read_csv(RAPs_filename,low_memory=False,index_col=0,usecols=[1,2])
    repeated_raps = raps[raps.Count>=10]
    
    raps_all_df.loc[raps.index,label_name]=[1 if df_prot.loc[df_prot[lbl]==True,rap].mean()>=df_prot.loc[df_prot[lbl]==False,rap].mean() else -1 for rap in raps.index]
    raps_all_df.loc[raps.index,'Any']=1
    
    repeated_raps_all_list.extend(repeated_raps.index.tolist())
    print(f'RAPs repeating 10+ times for {label_name}:')
    print(",\n".join(somalogic_names.loc[repeated_raps.index,'Target Name'].to_list()))
    

raps_all_df = raps_all_df.loc[raps_all_df['Any']==1,:].iloc[:,:-1]

raps_all_with_somalogic_data = somalogic_names.loc[somalogic_names.index.isin(raps_all_df.index.tolist()),soma_colnames]
raps_all_with_somalogic_data=raps_all_with_somalogic_data.rename({'Target Full Name':'Full Name'},axis=1)


raps_all_df2=pd.concat([raps_all_with_somalogic_data.loc[raps_all_df.index,'Target Name'] , raps_all_df],axis=1)
for col in raps_all_df2.columns[1:]:
    raps_all_df2[col]=raps_all_df2[col].apply(lambda val: 'AE-positive' if val==1 else 'AE-negative' if val==-1 else 'Irrelevant')

raps_all_with_somalogic_data.to_excel(f'RAPS_analysis/RAPs_with_names({"+".join(label_names)}).xlsx')
raps_all_df2.to_excel(f'RAPS_analysis/RAPs_with_direction({"+".join(label_names)}).xlsx')

pd.concat([raps_all_with_somalogic_data,raps_all_df2],axis=1).to_excel(f'RAPS_analysis/RAPs_full({"+".join(label_names)}).xlsx')

tmp = raps_all_df.abs().sum(axis=1).sort_values(ascending=False)
tmp7 = tmp[tmp>=7]
tmp6 = tmp[tmp>=6]
tmp5 = tmp[tmp>=5]
print('Proteins repeating in all 7 models:')
print("\n".join(somalogic_names.loc[tmp7.index,'Entrez Gene Name']))

print('ALL DONE!!!')