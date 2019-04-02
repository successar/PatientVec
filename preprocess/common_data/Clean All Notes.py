import pandas as pd
from PatientVec.preprocess.cleaners import cleaner_whatinnote
df = pd.read_csv('../../../bigdata/MIMIC/NOTEEVENTS.csv')

texts = list(df['TEXT'])
from tqdm import tqdm

for i in tqdm(range(len(texts)), mininterval=60) :
    texts[i] = cleaner_whatinnote(texts[i])
    
df['TEXT'] = texts

df = df.dropna(subset=['HADM_ID'])
df['CGID'] = df['CGID'].fillna(-1).astype(int)
df['HADM_ID'] = df['HADM_ID'].astype(int)
df['ISERROR'] = df['ISERROR'].apply(lambda s : '{0:.0f}'.format(s) if not pd.isna(s) else '')

df.reset_index(drop=True).to_csv('../../../bigdata/MIMIC/NOTEEVENTS_CLEANED.csv', index=False)

