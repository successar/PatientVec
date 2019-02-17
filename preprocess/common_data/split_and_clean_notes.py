import argparse
parser = argparse.ArgumentParser(description='Run Notes Splitting')
parser.add_argument("--iter", type=int, required=True)
parser.add_argument('--chunk', type=int, required=True)
parser.add_argument('--notes_file', type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)

import pandas as pd
from PatientVec.preprocess.cleaners import cleaner_whatinnote

i = 0
for chunk in pd.read_csv('../../bigdata/MIMIC/NOTEEVENTS.csv', chunksize=args.chunk) :
    i += 1
    if i == args.iter :
        texts = list(chunk['TEXT'])
        for j in range(len(texts)) :
            texts[j] = cleaner_whatinnote(texts[j])
            if j%1000 == 0:
                print(j)
                
        chunk['TEXT'] = texts
        chunk.to_csv(os.path.join(args.output_dir, 'notes_chunk_'+str(args.iter) + '.csv'))
    
        break
    
    


