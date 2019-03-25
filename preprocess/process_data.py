import pandas as pd
import os

import argparse
parser = argparse.ArgumentParser(description='Generate Dataset')
parser.add_argument('--main_file', type=str, required=True)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--embeddings_file', type=str)
parser.add_argument("--id_field", type=str)
parser.add_argument('--text_field', type=str)
parser.add_argument('--label_field', type=str)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

from PatientVec.preprocess.vocabulary import *
dataset = pd.read_csv(args.main_file)

from sklearn.model_selection import train_test_split

ID_field = args.id_field
text_field = args.text_field
label_field = args.label_field
structured_fields = [x for x in dataset.columns if x not in [ID_field, text_field, label_field]]

remain_idx, test_idx = train_test_split(list(dataset[ID_field]), test_size=0.2, stratify=list(dataset[label_field]), random_state=1389)
remain_labels = list(dataset[dataset[ID_field].isin(remain_idx)][label_field])
breakpoint()
train_idx, dev_idx = train_test_split(remain_idx, test_size=0.1, stratify=remain_labels, random_state=1389)

dataset['exp_split'] = None
dataset.loc[dataset[ID_field].isin(train_idx), ['exp_split']] = 'train'
dataset.loc[dataset[ID_field].isin(dev_idx), ['exp_split']] = 'dev'
dataset.loc[dataset[ID_field].isin(test_idx), ['exp_split']] = 'test'

dataset.drop(columns=[text_field]).to_msgpack(os.path.join(args.output_dir, 'Split_Structured_Final.msg'))

from PatientVec.preprocess.vocabulary import Vocabulary
vocab = Vocabulary(min_df=10)

texts = list(dataset[text_field])
assert type(texts[0]) == str
vocab.fit(texts)
vocab.save(args.output_dir)

from PatientVec.preprocess.embedder import PretrainedEmbedding
vocab = Vocabulary().load(args.output_dir)

embedder = PretrainedEmbedding(vocab)
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format(args.embeddings_file, binary=True)
embedder.extract_embeddings_from_gensim_model(model)
embedder.save(args.output_dir)

from PatientVec.preprocess.vectorizer import Vectorizer
vec = Vectorizer(vocab)
sequences = vec.texts_to_sequences(texts)

sequences = [[x] for x in sequences]
pickle.dump(sequences, open(os.path.join(args.output_dir, 'combined_notes_sequences.p'), 'wb'))
