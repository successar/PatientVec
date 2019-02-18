import argparse
parser = argparse.ArgumentParser(description='Run Notes Splitting')
parser.add_argument("--iter", type=int, required=True)
parser.add_argument('--chunk', type=int, required=True)
parser.add_argument('--notes_file', type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

import pandas as pd
from PatientVec.preprocess.cleaners import cleaner_whatinnote

i = 0
logging.info('Starting process')
for chunk in pd.read_csv(args.notes_file, chunksize=args.chunk) :
    i += 1
    logging.info('Read chunk %d', i)
    if i == args.iter :
        logging.info('processing chunk %d = %d', i, args.iter)
        texts = list(chunk['TEXT'])
        logging.info('Num text in chunk %d', len(texts))
        for j in range(len(texts)) :
            texts[j] = cleaner_whatinnote(texts[j])
            if j%1000 == 0:
                logging.info('Done iterations %d', j)
                
        chunk['TEXT'] = texts
        logging.info('Saving chunk to file')
        chunk.to_csv(os.path.join(args.output_dir, 'notes_chunk_'+str(args.iter) + '.csv'))
    
        break
        
logging.info('Done ..')
    
    


